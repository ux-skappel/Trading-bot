"""Strategy layer.

Translates ScoredOpportunity objects into actionable trade requests,
honoring the human-approval mode and the live-trading kill switch.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import List, Optional

from ..analysis.opportunity_scoring import ScoredOpportunity
from ..config import get_settings
from ..connectors.market_data import MarketDataProvider
from ..storage.database import session_scope
from ..storage.models import DecisionLog, OpportunityScore
from .broker_base import OrderRequest, OrderResult
from .paper_broker import PaperBroker
from .portfolio import Portfolio
from .risk_manager import RiskManager

logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    """Human-readable recommendation surfaced to the user."""

    symbol: str
    asset_class: str
    direction: str           # buy | avoid | watch
    confidence: float
    score: float
    reason: str
    sources: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    suggested_entry: Optional[float] = None
    suggested_stop_loss: Optional[float] = None
    suggested_take_profit: Optional[float] = None
    suggested_quantity: float = 0.0
    approval_pending: bool = True
    trade_id: Optional[int] = None


class TradingStrategy:
    """Glue between scoring, risk, broker and the recommendation queue."""

    def __init__(
        self,
        broker: Optional[PaperBroker] = None,
        portfolio: Optional[Portfolio] = None,
        risk: Optional[RiskManager] = None,
        market_data: Optional[MarketDataProvider] = None,
    ) -> None:
        self.settings = get_settings()
        self.market_data = market_data or MarketDataProvider()
        self.portfolio = portfolio or Portfolio(self.market_data)
        self.risk = risk or RiskManager(self.portfolio, self.market_data)
        self.broker = broker or PaperBroker(self.market_data)

    # --- public API ---

    def process(
        self,
        opportunity: ScoredOpportunity,
        source_urls: Optional[List[str]] = None,
    ) -> Recommendation:
        """Persist the opportunity, then route to risk + broker or queue it."""
        self._persist_opportunity(opportunity, source_urls or [])

        # SAFETY: never auto-execute in live mode unless live-trading is on AND
        # human-approval mode is off. In MVP we don't ship a live broker at all.
        live_active = self.settings.live_trading and self.settings.broker != "paper"
        if live_active:
            logger.warning(
                "LIVE trading flag is set but only a PaperBroker is wired in. "
                "Refusing to send live orders."
            )

        rec = self._to_recommendation(opportunity, source_urls or [])

        if opportunity.direction != "buy":
            self._log_skip(opportunity, "non-buy direction (watch/avoid/sell)")
            rec.approval_pending = False
            return rec

        quote = self.market_data.get_quote(opportunity.symbol, opportunity.asset_class)
        in_universe = self._in_universe(opportunity.symbol, opportunity.asset_class)
        is_blacklisted = opportunity.symbol in self.settings.blacklist

        risk_decision = self.risk.evaluate(
            symbol=opportunity.symbol,
            asset_class=opportunity.asset_class,
            score=opportunity.total_score,
            confidence=opportunity.confidence,
            quote=quote,
            is_blacklisted=is_blacklisted,
            is_in_universe=in_universe,
        )

        if not risk_decision.allowed:
            self._log_skip(opportunity, f"risk blocked: {risk_decision.reason}")
            rec.risk_factors.extend(risk_decision.blockers)
            rec.approval_pending = False
            rec.direction = "watch" if opportunity.total_score >= self.settings.min_score_to_recommend else "avoid"
            return rec

        rec.suggested_quantity = risk_decision.suggested_quantity

        req = OrderRequest(
            symbol=opportunity.symbol,
            asset_class=opportunity.asset_class,
            side="buy",
            quantity=risk_decision.suggested_quantity,
            stop_loss=opportunity.suggested_stop_loss,
            take_profit=opportunity.suggested_take_profit,
            trailing_stop_pct=(
                self.settings.trailing_stop_pct if self.settings.use_trailing_stop else None
            ),
            reason=" / ".join(opportunity.reasoning[-3:]),
        )

        # In paper mode + human_approval_mode -> creates a pending trade.
        result: OrderResult = self.broker.place_order(req, confirmation_token=None)
        rec.trade_id = result.trade_id
        rec.approval_pending = not result.accepted
        return rec

    # --- helpers ---

    def _in_universe(self, symbol: str, asset_class: str) -> bool:
        if asset_class == "crypto":
            return symbol in self.settings.crypto_watchlist
        return symbol in self.settings.stock_watchlist

    def _persist_opportunity(self, op: ScoredOpportunity, sources: List[str]) -> None:
        with session_scope() as db:
            db.add(
                OpportunityScore(
                    symbol=op.symbol,
                    asset_class=op.asset_class,
                    direction=op.direction,
                    total_score=op.total_score,
                    factors_json=json.dumps(op.factors),
                    reasoning="\n".join(op.reasoning),
                    sources_json=json.dumps(sources),
                    confidence=op.confidence,
                    risk_level=op.risk_level,
                )
            )

    def _log_skip(self, op: ScoredOpportunity, why: str) -> None:
        logger.info("Skipping %s: %s", op.symbol, why)
        with session_scope() as db:
            db.add(
                DecisionLog(
                    symbol=op.symbol,
                    action="skip",
                    decision="rejected",
                    score=op.total_score,
                    reason=why,
                    metadata_json=json.dumps(
                        {"direction": op.direction, "confidence": op.confidence}
                    ),
                )
            )

    def _to_recommendation(
        self, op: ScoredOpportunity, source_urls: List[str]
    ) -> Recommendation:
        return Recommendation(
            symbol=op.symbol,
            asset_class=op.asset_class,
            direction=op.direction,
            confidence=op.confidence,
            score=op.total_score,
            reason="\n".join(op.reasoning),
            sources=source_urls,
            risk_factors=[],
            suggested_entry=op.suggested_entry,
            suggested_stop_loss=op.suggested_stop_loss,
            suggested_take_profit=op.suggested_take_profit,
        )
