"""Intraday candles for replaying copied trades.

Hyperliquid first, since Liquid routes its perps there: crypto from the main
market, stocks/indices/commodities from the "xyz" market (xyz:GOOGL,
xyz:SP500, xyz:GOLD ...), which trade around the clock. Yahoo Finance is the
fallback. Hyperliquid only serves the most recent 5,000 candles per interval,
so the finest interval that still reaches back far enough is chosen.

Each market's max leverage is also exposed: Liquid's maintenance margin is
half the initial margin at that max (1.25% at 40x, 5% at 10x).
"""

from __future__ import annotations

import logging
import warnings
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import requests

from ..analysis.support_resistance import Bar
from .parser import KNOWN_COINS

logger = logging.getLogger(__name__)

HL_URL = "https://api.hyperliquid.xyz/info"
# Hyperliquid lists some coins per 1,000 units ("kPEPE"); prices are scaled back.
HL_THOUSANDS = {"PEPE", "BONK", "SHIB", "FLOKI", "LUNC", "NEIRO", "DOGS", "CAT"}
HL_RENAMES = {"MATIC": "POL", "RNDR": "RENDER"}
STOCK_DEX = "xyz"
# Yahoo tickers for names that aren't plain stock tickers (fallback only).
YAHOO_TICKERS = {"SP500": "^GSPC", "XYZ100": "^NDX", "GOLD": "GC=F", "SILVER": "SI=F",
                 "CL": "CL=F", "BRENTOIL": "BZ=F", "NATGAS": "NG=F", "COPPER": "HG=F"}
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
        self._markets: Optional[Dict[str, Dict[str, float]]] = None
        self._cache: Dict[Tuple[str, str, int, int], Tuple[List[Bar], str]] = {}

    def _hl_post(self, payload: dict):
        resp = self.http.post(HL_URL, json=payload, timeout=20)
        resp.raise_for_status()
        return resp.json()

    def markets(self) -> Dict[str, Dict[str, float]]:
        """{"main": {coin: max_lev}, "xyz": {"xyz:GOOGL": max_lev, ...}}."""
        if self._markets is None:
            self._markets = {}
            for dex in ("main", STOCK_DEX):
                payload = {"type": "meta"} if dex == "main" else {"type": "meta", "dex": dex}
                try:
                    meta = self._hl_post(payload)
                    self._markets[dex] = {a["name"]: float(a.get("maxLeverage") or 0)
                                          for a in meta.get("universe", []) if not a.get("isDelisted")}
                except Exception as exc:
                    logger.warning("Hyperliquid %s markets unavailable: %s", dex, exc)
                    self._markets[dex] = {}
        return self._markets

    def hl_universe(self) -> set:
        return set(self.markets().get("main", {}))

    def resolve(self, symbol: str) -> Tuple[Optional[str], float, Optional[float]]:
        """(Hyperliquid coin, price scale, max leverage) — coin None when not listed.

        Known crypto goes to the main market; anything else (stocks, indices,
        gold ...) to the xyz market first, so e.g. "SP500" never hits the SPX
        meme coin.
        """
        coin, scale = self.hl_coin(symbol)
        main, stocks = self.markets().get("main", {}), self.markets().get(STOCK_DEX, {})
        stock_coin = f"{STOCK_DEX}:{symbol.upper()}"
        is_crypto = symbol.upper() in KNOWN_COINS or symbol.upper() in HL_THOUSANDS
        if is_crypto and coin in main:
            return coin, scale, main[coin] or None
        if stock_coin in stocks:
            return stock_coin, 1.0, stocks[stock_coin] or None
        if coin in main:
            return coin, scale, main[coin] or None
        return None, 1.0, None

    def max_leverage(self, symbol: str) -> Optional[float]:
        return self.resolve(symbol)[2]

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

        coin, scale, _ = self.resolve(symbol)
        result: Tuple[List[Bar], str] = ([], "none")
        if coin is not None:
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
                    result = (bars, f"hyperliquid {coin} {interval}")
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
        if symbol.upper() in YAHOO_TICKERS:
            candidates = [YAHOO_TICKERS[symbol.upper()]]
        else:
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
                                                     end=end + timedelta(days=1), interval=interval,
                                                     prepost=True)
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
