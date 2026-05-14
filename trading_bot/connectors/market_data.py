"""Market data connector.

Uses yfinance for quotes/history. Falls back to a deterministic mock so the bot
remains demonstrable offline.
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

CRYPTO_YF_MAP = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD",
    "XRP": "XRP-USD",
    "DOGE": "DOGE-USD",
    "ADA": "ADA-USD",
    "AVAX": "AVAX-USD",
    "LINK": "LINK-USD",
}


@dataclass
class Quote:
    """A point-in-time market data snapshot."""

    symbol: str
    price: float
    prev_close: Optional[float] = None
    volume: Optional[float] = None
    avg_volume: Optional[float] = None
    daily_atr_pct: Optional[float] = None
    source: str = "yfinance"

    @property
    def daily_change_pct(self) -> float:
        if not self.prev_close:
            return 0.0
        return (self.price - self.prev_close) / self.prev_close

    @property
    def volume_change_pct(self) -> float:
        if not self.avg_volume:
            return 0.0
        return ((self.volume or 0) - self.avg_volume) / self.avg_volume


class MarketDataProvider:
    """Wrapper over yfinance with a deterministic offline mock fallback."""

    def __init__(self, provider: str = "yfinance") -> None:
        self.provider = provider

    def _to_yf_symbol(self, symbol: str, asset_class: str) -> str:
        if asset_class == "crypto":
            return CRYPTO_YF_MAP.get(symbol.upper(), f"{symbol.upper()}-USD")
        return symbol

    def get_quote(self, symbol: str, asset_class: str = "stock") -> Optional[Quote]:
        """Return a Quote for the symbol, or a deterministic mock if offline."""
        yf_symbol = self._to_yf_symbol(symbol, asset_class)
        try:
            import yfinance as yf  # type: ignore

            t = yf.Ticker(yf_symbol)
            hist = t.history(period="30d", interval="1d", auto_adjust=False)
            if hist is None or hist.empty:
                raise RuntimeError("empty history")
            last = hist.iloc[-1]
            prev = hist.iloc[-2] if len(hist) >= 2 else last
            high_low = (hist["High"] - hist["Low"]) / hist["Close"]
            atr_pct = float(high_low.tail(14).mean()) if not high_low.empty else 0.0
            return Quote(
                symbol=symbol,
                price=float(last["Close"]),
                prev_close=float(prev["Close"]),
                volume=float(last.get("Volume", 0.0)),
                avg_volume=float(hist["Volume"].tail(20).mean()) if "Volume" in hist else 0.0,
                daily_atr_pct=atr_pct,
                source="yfinance",
            )
        except Exception as exc:
            logger.info("market data offline for %s (%s); using mock", symbol, exc)
            return self._mock_quote(symbol)

    @staticmethod
    def _mock_quote(symbol: str) -> Quote:
        """Deterministic pseudo-random quote so demos work offline."""
        h = int(hashlib.md5(symbol.encode()).hexdigest(), 16)
        base = 10 + (h % 500)
        change = ((h >> 8) % 200 - 100) / 1000.0   # -10%..+10%
        prev = float(base)
        price = prev * (1 + change)
        vol = 1_000_000 + (h % 5_000_000)
        return Quote(
            symbol=symbol,
            price=round(price, 4),
            prev_close=round(prev, 4),
            volume=float(vol),
            avg_volume=float(vol) * 0.9,
            daily_atr_pct=round(abs(change) + 0.02, 4),
            source="mock",
        )

    def liquidity_ok(self, quote: Quote, asset_class: str) -> bool:
        """Coarse liquidity gate so we never paper-trade illiquid names."""
        if quote.source == "mock":
            return True  # never block demos
        if asset_class == "crypto":
            return (quote.volume or 0) * quote.price > 5_000_000
        return (quote.volume or 0) > 100_000

    @staticmethod
    def is_extreme_volatility(quote: Quote, threshold: float) -> bool:
        """Return True if daily ATR exceeds the configured threshold."""
        return (quote.daily_atr_pct or 0.0) > threshold
