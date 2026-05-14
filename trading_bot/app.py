"""Pipeline orchestrator: ingest -> normalize -> classify -> score -> trade."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from sqlalchemy import select

from .analysis.entity_extraction import Entities, extract_entities
from .analysis.event_classifier import EventLabel, classify_event
from .analysis.high_impact import (
    HighImpactClassification,
    classify_high_impact_post,
)
from .analysis.opportunity_scoring import ScoringInputs, score_opportunity
from .analysis.sentiment import is_rumor, score_sentiment
from .config import get_settings
from .connectors.base import IngestedItem
from .connectors.market_data import MarketDataProvider
from .connectors.reddit_api import RedditConnector
from .connectors.rss_feeds import RSSConnector
from .connectors.x_api import XConnector
from .storage.database import init_db, session_scope
from .storage.models import NormalizedEvent, RawEvent
from .trading.paper_broker import PaperBroker
from .trading.portfolio import Portfolio
from .trading.risk_manager import RiskManager
from .trading.strategy import Recommendation, TradingStrategy

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Summary of a single end-to-end run."""

    ingested: int = 0
    normalized: int = 0
    opportunities: int = 0
    recommendations: List[Recommendation] = field(default_factory=list)
    high_impact_hits: int = 0


class TradingBotApp:
    """End-to-end pipeline. One method, ``run_once()``, performs a full cycle."""

    def __init__(self) -> None:
        self.settings = get_settings()
        init_db()
        self.market_data = MarketDataProvider(self.settings.market_data_provider)
        self.portfolio = Portfolio(self.market_data)
        self.risk = RiskManager(self.portfolio, self.market_data)
        self.broker = PaperBroker(self.market_data)
        self.strategy = TradingStrategy(
            broker=self.broker,
            portfolio=self.portfolio,
            risk=self.risk,
            market_data=self.market_data,
        )
        self.connectors = [
            RSSConnector(),
            XConnector(),
            RedditConnector(),
        ]

    # ---- ingestion ----

    def ingest(self) -> List[IngestedItem]:
        items: List[IngestedItem] = []
        for c in self.connectors:
            try:
                items.extend(c.fetch())
            except Exception as exc:
                logger.warning("connector %s failed: %s", c.name, exc)
        logger.info("Ingested %d items total", len(items))
        return items

    def persist_raw(self, items: List[IngestedItem]) -> List[IngestedItem]:
        """Persist raw events and drop ones we've already seen."""
        fresh: List[IngestedItem] = []
        with session_scope() as db:
            seen = {row.dedupe_key for row in db.scalars(select(RawEvent))}
            for it in items:
                key = it.dedupe_key()
                if key in seen:
                    continue
                seen.add(key)
                db.add(
                    RawEvent(
                        source=it.source,
                        author=it.author,
                        url=it.url,
                        title=it.title,
                        text=it.text,
                        dedupe_key=key,
                    )
                )
                fresh.append(it)
        logger.info("Persisted %d new raw events", len(fresh))
        return fresh

    # ---- normalization ----

    def normalize(self, items: List[IngestedItem]) -> List[Tuple[NormalizedEvent, Entities, Optional[EventLabel], HighImpactClassification]]:
        out: List[Tuple[NormalizedEvent, Entities, Optional[EventLabel], HighImpactClassification]] = []
        with session_scope() as db:
            for it in items:
                ents = extract_entities(it.text)
                label = classify_event(it.text)
                hi = classify_high_impact_post(it.author, it.text)
                sentiment = score_sentiment(it.text)
                rumor = is_rumor(it.text)
                cred = self.settings.source_credibility.get(
                    it.source_domain or "", 0.5
                )
                # social media baseline credibility lower than news
                if it.source in {"reddit", "x"}:
                    cred = min(cred, 0.35)
                # high-impact official accounts override upward
                if hi.is_high_impact and hi.official_account:
                    cred = max(cred, hi.account_credibility)
                elif hi.is_high_impact:
                    cred = max(cred, 0.55)

                conf = (cred + (0 if rumor else 0.2)) / 2

                ne = NormalizedEvent(
                    timestamp=it.timestamp,
                    source=it.source,
                    source_domain=it.source_domain,
                    author=it.author,
                    text=it.text[:5000],
                    url=it.url,
                    companies=",".join(ents.companies),
                    tickers=",".join(ents.tickers),
                    crypto_symbols=",".join(ents.crypto),
                    event_type=label.event_type if label else None,
                    sentiment=sentiment,
                    is_rumor=rumor,
                    is_high_impact_account=hi.is_high_impact,
                    high_impact_topics=",".join(hi.detected_topics) if hi.is_high_impact else None,
                    confidence=conf,
                    dedupe_key=it.dedupe_key(),
                )
                db.add(ne)
                db.flush()
                db.expunge(ne)
                out.append((ne, ents, label, hi))
        logger.info("Normalized %d events", len(out))
        return out

    # ---- scoring + dispatch ----

    def score_and_dispatch(
        self,
        normalized: List[Tuple[NormalizedEvent, Entities, Optional[EventLabel], HighImpactClassification]],
    ) -> List[Recommendation]:
        # Group by (symbol, asset_class) to count duplicate confirmations.
        groups: Dict[Tuple[str, str], List[Tuple[NormalizedEvent, Entities, Optional[EventLabel], HighImpactClassification]]] = {}
        for ne, ents, label, hi in normalized:
            symbols: List[Tuple[str, str]] = []
            for t in ents.tickers:
                symbols.append((t, "stock"))
            for c in ents.crypto:
                symbols.append((c, "crypto"))
            # Add high-impact derived universe
            if hi.is_high_impact:
                for t in hi.affected_stocks:
                    symbols.append((t, "stock"))
                for c in hi.affected_crypto:
                    symbols.append((c, "crypto"))
            for sym, cls in set(symbols):
                groups.setdefault((sym, cls), []).append((ne, ents, label, hi))

        recommendations: List[Recommendation] = []
        for (symbol, asset_class), evidence in groups.items():
            # take the most recent + use independent-source count
            evidence.sort(key=lambda x: x[0].timestamp, reverse=True)
            ne, ents, label, hi = evidence[0]
            independent_sources = len({e[0].source_domain or e[0].source for e in evidence})
            duplicate_confirmations = max(0, independent_sources - 1)
            quote = self.market_data.get_quote(symbol, asset_class)
            in_universe = (
                symbol in self.settings.stock_watchlist
                if asset_class == "stock"
                else symbol in self.settings.crypto_watchlist
            )
            scoring = ScoringInputs(
                symbol=symbol,
                asset_class=asset_class,
                sentiment=ne.sentiment,
                source_credibility=ne.confidence,
                event_timestamp=ne.timestamp,
                event_label=label,
                high_impact=hi if hi.is_high_impact else None,
                quote=quote,
                duplicate_confirmations=duplicate_confirmations,
                is_in_universe=in_universe,
                is_blacklisted=symbol in self.settings.blacklist,
                is_rumor=ne.is_rumor,
            )
            scored = score_opportunity(scoring)
            # Prepend high-impact reasoning if present
            if hi.is_high_impact:
                scored.reasoning = hi.reasoning + scored.reasoning
            source_urls = [e[0].url for e in evidence if e[0].url][:5]
            rec = self.strategy.process(scored, source_urls=source_urls)
            recommendations.append(rec)
        return recommendations

    # ---- one-shot ----

    def run_once(self) -> PipelineResult:
        result = PipelineResult()
        items = self.ingest()
        result.ingested = len(items)
        fresh = self.persist_raw(items)
        normalized = self.normalize(fresh)
        result.normalized = len(normalized)
        result.high_impact_hits = sum(1 for _, _, _, hi in normalized if hi.is_high_impact)
        recs = self.score_and_dispatch(normalized)
        result.opportunities = len(recs)
        result.recommendations = recs
        # update open positions (stop/take/trailing)
        self.broker.update_open_positions()
        self.portfolio.snapshot()
        return result
