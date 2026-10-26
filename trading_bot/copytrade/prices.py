"""Intraday candles for replaying copied trades.

Hyperliquid first (Liquid routes its perps there, so these are the prices a
copy would actually get), Yahoo Finance as the fallback (e.g. stocks).
Hyperliquid only serves the most recent 5,000 candles per interval, so the
finest interval that still reaches back far enough is chosen automatically.
"""

from __future__ import annotations

import logging
import warnings
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import requests

from ..analysis.support_resistance import Bar

logger = logging.getLogger(__name__)

HL_URL = "https://api.hyperliquid.xyz/info"
# Hyperliquid lists some coins per 1,000 units ("kPEPE"); prices are scaled back.
HL_THOUSANDS = {"PEPE", "BONK", "SHIB", "FLOKI", "LUNC", "NEIRO", "DOGS", "CAT"}
HL_RENAMES = {"MATIC": "POL", "RNDR": "RENDER"}
INTERVALS: List[Tuple[str, int]] = [("1m", 60), ("5m", 300), ("15m", 900), ("1h", 3600), ("4h", 14400)]
MAX_CANDLES = 4900          # stay under Hyperliquid's 5,000 limit


def _utc_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def pick_interval(start: datetime, now: Optional[datetime] = None) -> Tuple[str, int]:
    """Finest interval whose last 5,000 candles still reach back to ``start``."""
    now = now or datetime.utcnow()
    span = (now - start).total_seconds()
    for name, seconds in INTERVALS:
        if span <= MAX_CANDLES * seconds:
            return name, seconds
    return INTERVALS[-1]


class CandleFeed:
    """Fetches and caches candles per symbol."""

    def __init__(self, session: Optional[requests.Session] = None) -> None:
        self.http = session or requests.Session()
        self._universe: Optional[set] = None
        self._cache: Dict[Tuple[str, str, int, int], Tuple[List[Bar], str]] = {}

    def _hl_post(self, payload: dict):
        resp = self.http.post(HL_URL, json=payload, timeout=20)
        resp.raise_for_status()
        return resp.json()

    def hl_universe(self) -> set:
        if self._universe is None:
            try:
                meta = self._hl_post({"type": "meta"})
                self._universe = {asset["name"] for asset in meta.get("universe", [])}
            except Exception as exc:
                logger.warning("Hyperliquid meta unavailable: %s", exc)
                self._universe = set()
        return self._universe

    @staticmethod
    def hl_coin(symbol: str) -> Tuple[str, float]:
        sym = HL_RENAMES.get(symbol.upper(), symbol.upper())
        if sym in HL_THOUSANDS:
            return f"k{sym}", 1000.0
        return sym, 1.0

    def candles(self, symbol: str, start: datetime, end: Optional[datetime] = None) -> Tuple[List[Bar], str]:
        """Candles covering [start, end] (oldest first) and a source label."""
        now = datetime.utcnow()
        end = min(end or now, now)
        interval, seconds = pick_interval(start, now)
        key = (symbol.upper(), interval, _utc_ms(start) // 60000, _utc_ms(end) // 60000)
        if key in self._cache:
            return self._cache[key]

        coin, scale = self.hl_coin(symbol)
        result: Tuple[List[Bar], str] = ([], "none")
        if coin in self.hl_universe():
            try:
                rows = self._hl_post({"type": "candleSnapshot", "req": {
                    "coin": coin, "interval": interval,
                    "startTime": _utc_ms(start - timedelta(seconds=seconds)), "endTime": _utc_ms(end)}})
                bars = [Bar(
                    timestamp=datetime.utcfromtimestamp(r["t"] / 1000),
                    open=float(r["o"]) / scale, high=float(r["h"]) / scale,
                    low=float(r["l"]) / scale, close=float(r["c"]) / scale,
                    volume=float(r["v"]),
                ) for r in rows]
                if bars:
                    result = (bars, f"hyperliquid {interval}")
            except Exception as exc:
                logger.warning("Hyperliquid candles failed for %s: %s", symbol, exc)
        if not result[0]:
            result = self._yahoo(symbol, start, end)
        self._cache[key] = result
        return result

    @staticmethod
    def _yahoo(symbol: str, start: datetime, end: datetime) -> Tuple[List[Bar], str]:
        span = datetime.utcnow() - start
        interval = "1m" if span < timedelta(days=7) else "5m" if span < timedelta(days=59) else \
            "1h" if span < timedelta(days=729) else "1d"
        candidates = [symbol, f"{symbol}-USD"] if "-" not in symbol else [symbol]
        try:
            import yfinance as yf  # type: ignore
        except ImportError:
            return [], "none"
        for ticker in candidates:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    hist = yf.Ticker(ticker).history(start=start - timedelta(days=1),
                                                     end=end + timedelta(days=1), interval=interval)
                if hist is None or hist.empty:
                    continue
                bars = []
                for ts, row in hist.iterrows():
                    stamp = ts.to_pydatetime()
                    if stamp.tzinfo is not None:
                        stamp = stamp.astimezone(timezone.utc).replace(tzinfo=None)
                    bars.append(Bar(stamp, float(row["Open"]), float(row["High"]),
                                    float(row["Low"]), float(row["Close"]), float(row.get("Volume", 0) or 0)))
                return bars, f"yahoo {interval}"
            except Exception as exc:
                logger.info("Yahoo candles failed for %s: %s", ticker, exc)
        return [], "none"
