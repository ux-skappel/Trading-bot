"""Market data connector.

Uses yfinance for daily history and quotes. Falls back to a deterministic
mock so the bot remains demonstrable offline.
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from ..analysis.support_resistance import Bar

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

DEFAULT_PERIOD = "6mo"
DEFAULT_INTERVAL = "1d"


def infer_asset_class(symbol: str) -> str:
    """Guess whether a symbol is crypto or a stock."""
    from ..config import get_settings

    sym = symbol.upper()
    if sym in CRYPTO_YF_MAP or sym.endswith("-USD") or sym in get_settings().crypto_watchlist:
        return "crypto"
    return "stock"


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


@dataclass
class PriceHistory:
    """OHLCV bars for one symbol, oldest first."""

    symbol: str
    bars: List[Bar]
    source: str


class MarketDataProvider:
    """Wrapper over yfinance with a deterministic offline mock fallback."""

    def __init__(self, provider: str = "yfinance", cache_ttl_seconds: float = 120.0) -> None:
        self.provider = provider
        self.cache_ttl = cache_ttl_seconds
        self._cache: Dict[Tuple[str, str, str, str], Tuple[float, PriceHistory]] = {}

    @staticmethod
    def _to_yf_symbol(symbol: str, asset_class: str) -> str:
        if asset_class == "crypto" and not symbol.upper().endswith("-USD"):
            return CRYPTO_YF_MAP.get(symbol.upper(), f"{symbol.upper()}-USD")
        return symbol

    def get_history(
        self,
        symbol: str,
        asset_class: str = "stock",
        period: str = DEFAULT_PERIOD,
        interval: str = DEFAULT_INTERVAL,
    ) -> PriceHistory:
        """Return daily OHLCV history, or a deterministic mock if offline."""
        key = (symbol, asset_class, period, interval)
        cached = self._cache.get(key)
        if cached and time.monotonic() - cached[0] < self.cache_ttl:
            return cached[1]
        try:
            history = PriceHistory(symbol, self._fetch_yfinance(symbol, asset_class, period, interval), "yfinance")
        except Exception as exc:
            logger.info("market data offline for %s (%s); using mock", symbol, exc)
            history = PriceHistory(symbol, self._mock_history(symbol), "mock")
        self._cache[key] = (time.monotonic(), history)
        return history

    def _fetch_yfinance(self, symbol: str, asset_class: str, period: str, interval: str) -> List[Bar]:
        import yfinance as yf  # type: ignore

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # yfinance's pandas deprecation noise
            hist = yf.Ticker(self._to_yf_symbol(symbol, asset_class)).history(
                period=period, interval=interval, auto_adjust=False
            )
        if hist is None or hist.empty:
            raise RuntimeError("empty history")
        bars: List[Bar] = []
        for ts, row in hist.iterrows():
            values = [float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])]
            if any(math.isnan(v) for v in values):
                continue
            volume = float(row.get("Volume", 0.0) or 0.0)
            bars.append(Bar(ts.to_pydatetime().replace(tzinfo=None), *values, volume=volume))
        if not bars:
            raise RuntimeError("history has no valid rows")
        return bars

    def get_quote(self, symbol: str, asset_class: str = "stock") -> Optional[Quote]:
        """Return the latest Quote derived from daily history."""
        history = self.get_history(symbol, asset_class)
        bars = history.bars
        if not bars:
            return None
        last = bars[-1]
        prev = bars[-2] if len(bars) >= 2 else last
        ranges = [(b.high - b.low) / b.close for b in bars[-14:] if b.close]
        recent_volume = [b.volume for b in bars[-20:]]
        return Quote(
            symbol=symbol,
            price=last.close,
            prev_close=prev.close,
            volume=last.volume,
            avg_volume=sum(recent_volume) / len(recent_volume),
            daily_atr_pct=sum(ranges) / len(ranges) if ranges else 0.0,
            source=history.source,
        )

    @staticmethod
    def _mock_history(symbol: str, n: int = 126) -> List[Bar]:
        """Deterministic random-walk candles so demos work offline."""
        seed = int(hashlib.md5(symbol.encode()).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)
        price = float(10 + seed % 500)
        base_volume = 1_000_000 + seed % 5_000_000
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        bars: List[Bar] = []
        for i in range(n):
            open_ = price
            close = max(0.01, open_ * (1 + rng.gauss(0, 0.02)))
            high = max(open_, close) * (1 + abs(rng.gauss(0, 0.008)))
            low = min(open_, close) * (1 - abs(rng.gauss(0, 0.008)))
            bars.append(Bar(
                timestamp=today - timedelta(days=n - 1 - i),
                open=round(open_, 4),
                high=round(high, 4),
                low=round(low, 4),
                close=round(close, 4),
                volume=float(round(base_volume * (0.6 + rng.random() * 0.8))),
            ))
            price = close
        return bars

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
