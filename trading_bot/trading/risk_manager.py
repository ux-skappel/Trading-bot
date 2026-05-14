"""Risk manager — enforces hard limits before any trade is placed.

All decisions (allow/deny) are logged to the decision_logs table.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

from sqlalchemy import select

from ..config import get_settings
from ..connectors.market_data import MarketDataProvider, Quote
from ..storage.database import session_scope
from ..storage.models import DecisionLog, PaperTrade
from .portfolio import Portfolio

logger = logging.getLogger(__name__)


@dataclass
class RiskDecision:
    """Risk manager verdict for a single proposed trade."""

    allowed: bool
    reason: str
    suggested_quantity: float = 0.0
    suggested_notional: float = 0.0
    blockers: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.blockers is None:
            self.blockers = []


class RiskManager:
    """Checks proposed trades against configured limits."""

    def __init__(
        self,
        portfolio: Optional[Portfolio] = None,
        market_data: Optional[MarketDataProvider] = None,
    ) -> None:
        self.settings = get_settings()
        self.portfolio = portfolio or Portfolio()
        self.market_data = market_data or self.portfolio.market_data

    def _log(self, symbol: Optional[str], action: str, decision: str,
             reason: str, score: Optional[float] = None, **meta) -> None:
        with session_scope() as db:
            db.add(
                DecisionLog(
                    symbol=symbol,
                    action=action,
                    decision=decision,
                    reason=reason,
                    score=score,
                    metadata_json=json.dumps(meta) if meta else None,
                )
            )

    def evaluate(
        self,
        symbol: str,
        asset_class: str,
        score: float,
        confidence: float,
        quote: Optional[Quote],
        is_blacklisted: bool,
        is_in_universe: bool,
    ) -> RiskDecision:
        """Check every gate. Return RiskDecision with `allowed=False` on any block."""
        blockers: List[str] = []
        s = self.settings

        # Universe & blacklist
        if is_blacklisted:
            blockers.append("asset is blacklisted")
        if not is_in_universe:
            blockers.append("asset not in configured universe/watchlist")

        # Score / credibility gates
        if score < s.min_score_to_trade:
            blockers.append(f"score {score:.1f} below trade threshold {s.min_score_to_trade}")
        if confidence < s.min_credibility_to_trade:
            blockers.append(
                f"confidence {confidence:.2f} below threshold {s.min_credibility_to_trade}"
            )

        # Market data sanity
        if quote is None:
            blockers.append("no market data available")
        else:
            if not self.market_data.liquidity_ok(quote, asset_class):
                blockers.append("liquidity insufficient")
            if (
                self.market_data.is_extreme_volatility(quote, s.extreme_volatility_pct)
                and not s.allow_extreme_volatility
            ):
                blockers.append(
                    f"extreme volatility (daily ATR {quote.daily_atr_pct:.2%})"
                )

        # Portfolio-level gates
        snapshot = self.portfolio.snapshot()
        if len(snapshot.open_positions) >= s.max_open_positions:
            blockers.append(
                f"max open positions reached ({s.max_open_positions})"
            )

        # Duplicate position check
        if any(p.symbol == symbol for p in snapshot.open_positions):
            blockers.append("duplicate open position already exists")

        # Daily loss circuit breaker
        starting = max(1.0, s.starting_cash)
        if snapshot.daily_pnl / starting <= -s.max_daily_loss_pct:
            blockers.append(
                f"max daily loss hit ({snapshot.daily_pnl / starting:.2%})"
            )

        # Cooldown after a loss
        cooldown_start = datetime.utcnow() - timedelta(minutes=s.cooldown_minutes_after_loss)
        with session_scope() as db:
            recent_losses: List[PaperTrade] = list(
                db.scalars(
                    select(PaperTrade).where(
                        PaperTrade.symbol == symbol,
                        PaperTrade.status == "closed",
                        PaperTrade.closed_at >= cooldown_start,
                        PaperTrade.pnl < 0,
                    )
                )
            )
        if recent_losses:
            blockers.append(
                f"cooldown after recent loss on {symbol} "
                f"({s.cooldown_minutes_after_loss}min)"
            )

        # Asset-class exposure
        exposure = self.portfolio.exposure_by_class()
        class_exposure_pct = (
            exposure.get(asset_class, 0.0) / max(1.0, snapshot.total_value)
        )
        if class_exposure_pct >= s.max_exposure_per_class_pct:
            blockers.append(
                f"max {asset_class} class exposure reached "
                f"({class_exposure_pct:.0%})"
            )

        # Position sizing
        target_notional = snapshot.total_value * s.max_position_pct
        target_qty = 0.0
        if quote and quote.price > 0:
            target_qty = round(target_notional / quote.price, 6)
            if target_qty <= 0:
                blockers.append("calculated quantity is zero")

        allowed = not blockers
        reason = "ok" if allowed else "; ".join(blockers)
        self._log(
            symbol=symbol,
            action="risk_check",
            decision="approved" if allowed else "rejected",
            reason=reason,
            score=score,
            asset_class=asset_class,
            confidence=confidence,
            target_qty=target_qty,
            target_notional=target_notional,
        )
        return RiskDecision(
            allowed=allowed,
            reason=reason,
            suggested_quantity=target_qty,
            suggested_notional=target_notional,
            blockers=blockers,
        )
