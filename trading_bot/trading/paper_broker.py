"""In-memory + DB-backed paper broker. Never touches a real exchange."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import select

from ..config import get_settings
from ..connectors.market_data import MarketDataProvider
from ..storage.database import session_scope
from ..storage.models import DecisionLog, PaperTrade
from .broker_base import BrokerBase, OrderRequest, OrderResult

logger = logging.getLogger(__name__)


class PaperBroker(BrokerBase):
    """Simulated broker — fills at the current mock/live quote with zero fees."""

    name = "paper"
    supports_live = False

    def __init__(self, market_data: Optional[MarketDataProvider] = None) -> None:
        self.market_data = market_data or MarketDataProvider()

    def place_order(
        self,
        req: OrderRequest,
        confirmation_token: Optional[str] = None,
    ) -> OrderResult:
        settings = get_settings()
        # Even in paper mode, refuse to silently execute when human approval is
        # required and no confirmation_token has been supplied.
        if settings.human_approval_mode and not confirmation_token:
            return self._pending(req, "human_approval_mode is on; created pending trade")

        quote = self.market_data.get_quote(req.symbol, req.asset_class)
        if quote is None:
            return OrderResult(
                accepted=False,
                broker=self.name,
                symbol=req.symbol,
                side=req.side,
                quantity=req.quantity,
                fill_price=None,
                message="no market data",
            )

        fill_price = req.limit_price or quote.price
        with session_scope() as db:
            trade = PaperTrade(
                symbol=req.symbol,
                asset_class=req.asset_class,
                side=req.side,
                quantity=float(req.quantity),
                entry_price=float(fill_price),
                stop_loss=req.stop_loss,
                take_profit=req.take_profit,
                trailing_stop_pct=req.trailing_stop_pct,
                high_watermark=float(fill_price),
                status="open",
                reasoning=req.reason,
                approved_by=confirmation_token or "auto",
            )
            db.add(trade)
            db.flush()
            db.add(
                DecisionLog(
                    symbol=req.symbol,
                    action=req.side,
                    decision="simulated",
                    reason=req.reason or "paper trade opened",
                    metadata_json=json.dumps(
                        {"qty": req.quantity, "price": fill_price}
                    ),
                )
            )
            trade_id = trade.id

        logger.info(
            "[PAPER] %s %s qty=%s @ %.4f (trade #%s)",
            req.side.upper(),
            req.symbol,
            req.quantity,
            fill_price,
            trade_id,
        )
        return OrderResult(
            accepted=True,
            broker=self.name,
            symbol=req.symbol,
            side=req.side,
            quantity=req.quantity,
            fill_price=fill_price,
            message="paper trade opened",
            trade_id=trade_id,
        )

    def _pending(self, req: OrderRequest, msg: str) -> OrderResult:
        with session_scope() as db:
            trade = PaperTrade(
                symbol=req.symbol,
                asset_class=req.asset_class,
                side=req.side,
                quantity=float(req.quantity),
                entry_price=0.0,
                stop_loss=req.stop_loss,
                take_profit=req.take_profit,
                trailing_stop_pct=req.trailing_stop_pct,
                status="pending_approval",
                reasoning=req.reason,
            )
            db.add(trade)
            db.flush()
            db.add(
                DecisionLog(
                    symbol=req.symbol,
                    action=req.side,
                    decision="pending",
                    reason=msg,
                )
            )
            trade_id = trade.id
        return OrderResult(
            accepted=False,
            broker=self.name,
            symbol=req.symbol,
            side=req.side,
            quantity=req.quantity,
            fill_price=None,
            message=msg,
            trade_id=trade_id,
        )

    def cancel_order(self, trade_id: int) -> bool:
        with session_scope() as db:
            t = db.get(PaperTrade, trade_id)
            if not t or t.status not in {"open", "pending_approval"}:
                return False
            t.status = "cancelled"
            return True

    def get_position(self, symbol: str) -> Optional[dict]:
        with session_scope() as db:
            t = db.scalars(
                select(PaperTrade).where(
                    PaperTrade.symbol == symbol, PaperTrade.status == "open"
                )
            ).first()
            if not t:
                return None
            return {
                "id": t.id,
                "symbol": t.symbol,
                "qty": t.quantity,
                "entry_price": t.entry_price,
                "stop_loss": t.stop_loss,
                "take_profit": t.take_profit,
            }

    def approve_pending(self, trade_id: int, approver: str = "user") -> OrderResult:
        """Convert a pending_approval trade into an open trade at current price."""
        with session_scope() as db:
            t = db.get(PaperTrade, trade_id)
            if not t or t.status != "pending_approval":
                return OrderResult(
                    accepted=False, broker=self.name,
                    symbol=(t.symbol if t else ""), side=(t.side if t else ""),
                    quantity=(t.quantity if t else 0.0), fill_price=None,
                    message="no pending trade with that id",
                )
            quote = self.market_data.get_quote(t.symbol, t.asset_class)
            if quote is None:
                return OrderResult(
                    accepted=False, broker=self.name, symbol=t.symbol, side=t.side,
                    quantity=t.quantity, fill_price=None, message="no market data",
                )
            t.entry_price = float(quote.price)
            t.high_watermark = float(quote.price)
            t.status = "open"
            t.approved_by = approver
            db.add(
                DecisionLog(
                    symbol=t.symbol, action=t.side, decision="approved",
                    reason=f"approved by {approver}",
                )
            )
            return OrderResult(
                accepted=True, broker=self.name, symbol=t.symbol, side=t.side,
                quantity=t.quantity, fill_price=t.entry_price,
                message="approved & opened", trade_id=t.id,
            )

    def update_open_positions(self) -> None:
        """Walk open positions, apply stop loss / take profit / trailing stop."""
        with session_scope() as db:
            opens = list(
                db.scalars(select(PaperTrade).where(PaperTrade.status == "open"))
            )
            for t in opens:
                quote = self.market_data.get_quote(t.symbol, t.asset_class)
                if not quote:
                    continue
                price = quote.price
                exit_reason: Optional[str] = None

                if t.side == "buy":
                    if t.high_watermark is None or price > t.high_watermark:
                        t.high_watermark = price
                    if t.stop_loss and price <= t.stop_loss:
                        exit_reason = "stop_loss"
                    elif t.take_profit and price >= t.take_profit:
                        exit_reason = "take_profit"
                    elif t.trailing_stop_pct and t.high_watermark:
                        trail_level = t.high_watermark * (1 - t.trailing_stop_pct)
                        if price <= trail_level:
                            exit_reason = "trailing_stop"
                else:  # sell / short — not auto-opened in MVP but supported.
                    if t.high_watermark is None or price < t.high_watermark:
                        t.high_watermark = price
                    if t.stop_loss and price >= t.stop_loss:
                        exit_reason = "stop_loss"
                    elif t.take_profit and price <= t.take_profit:
                        exit_reason = "take_profit"

                if exit_reason:
                    pnl = (price - t.entry_price) * t.quantity
                    if t.side == "sell":
                        pnl = -pnl
                    t.exit_price = price
                    t.exit_reason = exit_reason
                    t.pnl = round(pnl, 4)
                    t.closed_at = datetime.utcnow()
                    t.status = "closed"
                    db.add(
                        DecisionLog(
                            symbol=t.symbol, action="close",
                            decision="simulated",
                            reason=f"{exit_reason} @ {price:.4f}, pnl={pnl:.2f}",
                        )
                    )
                    logger.info(
                        "[PAPER] CLOSE %s @ %.4f (%s, pnl=%.2f)",
                        t.symbol, price, exit_reason, pnl,
                    )
