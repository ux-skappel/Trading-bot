"""Portfolio bookkeeping for paper trading."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy import select

from ..config import get_settings
from ..connectors.market_data import MarketDataProvider
from ..storage.database import session_scope
from ..storage.models import PaperTrade, PortfolioSnapshot

logger = logging.getLogger(__name__)


@dataclass
class PortfolioState:
    """In-memory snapshot of the paper portfolio."""

    cash: float
    positions_value: float
    total_value: float
    open_positions: List[PaperTrade]
    daily_pnl: float


class Portfolio:
    """Computes and persists portfolio state."""

    def __init__(self, market_data: Optional[MarketDataProvider] = None) -> None:
        self.market_data = market_data or MarketDataProvider()
        self.settings = get_settings()

    def _starting_cash(self) -> float:
        return self.settings.starting_cash

    def cash_balance(self) -> float:
        """Cash = starting cash - cost of open positions - realized PnL adjustments."""
        with session_scope() as db:
            trades: List[PaperTrade] = list(db.scalars(select(PaperTrade)))
            spent_on_open = sum(
                t.entry_price * t.quantity for t in trades if t.status == "open"
            )
            realized = sum(
                (t.pnl or 0.0) for t in trades if t.status == "closed"
            )
        return self._starting_cash() - spent_on_open + realized

    def open_positions(self) -> List[PaperTrade]:
        with session_scope() as db:
            rows: List[PaperTrade] = list(
                db.scalars(select(PaperTrade).where(PaperTrade.status == "open"))
            )
            # detach
            for r in rows:
                db.expunge(r)
            return rows

    def positions_value(self, positions: Optional[List[PaperTrade]] = None) -> float:
        positions = positions if positions is not None else self.open_positions()
        total = 0.0
        for p in positions:
            q = self.market_data.get_quote(p.symbol, p.asset_class)
            price = q.price if q else p.entry_price
            total += price * p.quantity
        return total

    def daily_pnl(self) -> float:
        """Sum of realized + unrealized PnL since UTC midnight."""
        midnight = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        with session_scope() as db:
            closed_today: List[PaperTrade] = list(
                db.scalars(
                    select(PaperTrade).where(
                        PaperTrade.status == "closed",
                        PaperTrade.closed_at >= midnight,
                    )
                )
            )
            opens: List[PaperTrade] = list(
                db.scalars(select(PaperTrade).where(PaperTrade.status == "open"))
            )
            realized = sum((t.pnl or 0.0) for t in closed_today)
            unrealized = 0.0
            for p in opens:
                q = self.market_data.get_quote(p.symbol, p.asset_class)
                if q:
                    unrealized += (q.price - p.entry_price) * p.quantity
        return realized + unrealized

    def snapshot(self) -> PortfolioState:
        opens = self.open_positions()
        pv = self.positions_value(opens)
        cash = self.cash_balance()
        total = cash + pv
        dpnl = self.daily_pnl()
        with session_scope() as db:
            db.add(
                PortfolioSnapshot(
                    cash=cash,
                    positions_value=pv,
                    total_value=total,
                    daily_pnl=dpnl,
                    open_positions=len(opens),
                )
            )
        return PortfolioState(
            cash=cash,
            positions_value=pv,
            total_value=total,
            open_positions=opens,
            daily_pnl=dpnl,
        )

    def exposure_by_class(self) -> Dict[str, float]:
        """Return current $ exposure per asset_class."""
        out: Dict[str, float] = {}
        for p in self.open_positions():
            out[p.asset_class] = out.get(p.asset_class, 0.0) + p.entry_price * p.quantity
        return out
