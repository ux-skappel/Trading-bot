"""Broker interface.

Concrete brokers MUST inherit BrokerBase. Live brokers MUST refuse to place
orders unless LIVE_TRADING=true AND a manual confirmation token is provided.
For the MVP we ship only PaperBroker; the architecture is deliberately
structured so live trading cannot be accidentally enabled.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class OrderRequest:
    """A requested order. Stop/take are optional."""

    symbol: str
    asset_class: str
    side: str            # "buy" | "sell"
    quantity: float
    limit_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop_pct: Optional[float] = None
    reason: Optional[str] = None


@dataclass
class OrderResult:
    """Result of an order placement attempt."""

    accepted: bool
    broker: str
    symbol: str
    side: str
    quantity: float
    fill_price: Optional[float]
    message: str
    trade_id: Optional[int] = None


class BrokerBase(ABC):
    """Abstract broker interface."""

    name: str = "base"
    supports_live: bool = False

    @abstractmethod
    def place_order(self, req: OrderRequest, confirmation_token: Optional[str] = None) -> OrderResult:
        ...

    @abstractmethod
    def cancel_order(self, trade_id: int) -> bool:
        ...

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[dict]:
        ...
