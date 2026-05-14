"""Simple event-replay backtester.

Inputs (both CSV):
  events.csv   columns: timestamp, source, author, text, url
  prices.csv   columns: timestamp, symbol, price

For each event we score it, and for any "buy" recommendation we open a
simulated position at the next available price and close it after a
configurable horizon (default 5 bars) or on stop/take.

Metrics: win rate, average return, max drawdown, Sharpe-ish, n_trades,
best/worst trade.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import math

from ..analysis.entity_extraction import extract_entities
from ..analysis.event_classifier import classify_event
from ..analysis.high_impact import classify_high_impact_post
from ..analysis.opportunity_scoring import ScoringInputs, score_opportunity
from ..analysis.sentiment import is_rumor, score_sentiment
from ..config import get_settings
from ..connectors.market_data import Quote

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """Single backtested trade record."""

    symbol: str
    entry_time: datetime
    exit_time: Optional[datetime]
    entry: float
    exit: Optional[float]
    pnl_pct: Optional[float]
    reason: str


@dataclass
class BacktestResult:
    """Summary stats for a backtest run."""

    n_trades: int
    win_rate: float
    avg_return: float
    max_drawdown: float
    sharpe: float
    best_trade: Optional[BacktestTrade]
    worst_trade: Optional[BacktestTrade]
    trades: List[BacktestTrade]


def _load_csv(path: str) -> List[Dict[str, str]]:
    import csv
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def _parse_ts(s: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            continue
    return datetime.utcnow()


def _build_price_index(rows: List[Dict[str, str]]) -> Dict[str, List[Tuple[datetime, float]]]:
    out: Dict[str, List[Tuple[datetime, float]]] = {}
    for r in rows:
        sym = r["symbol"].strip().upper()
        ts = _parse_ts(r["timestamp"])
        out.setdefault(sym, []).append((ts, float(r["price"])))
    for k in out:
        out[k].sort(key=lambda x: x[0])
    return out


def _next_price(
    index: Dict[str, List[Tuple[datetime, float]]],
    symbol: str,
    after: datetime,
) -> Optional[Tuple[datetime, float]]:
    series = index.get(symbol, [])
    for ts, px in series:
        if ts >= after:
            return ts, px
    return None


def _price_at_or_before(
    index: Dict[str, List[Tuple[datetime, float]]],
    symbol: str,
    when: datetime,
) -> Optional[float]:
    series = index.get(symbol, [])
    last = None
    for ts, px in series:
        if ts <= when:
            last = px
        else:
            break
    return last


def run_backtest(
    events_csv: str,
    prices_csv: str,
    horizon_bars: int = 5,
    stop_loss_pct: Optional[float] = None,
    take_profit_pct: Optional[float] = None,
) -> BacktestResult:
    """Replay events against prices and return summary statistics."""
    settings = get_settings()
    stop = stop_loss_pct if stop_loss_pct is not None else settings.default_stop_loss_pct
    take = take_profit_pct if take_profit_pct is not None else settings.default_take_profit_pct

    events = _load_csv(events_csv)
    prices = _build_price_index(_load_csv(prices_csv))

    trades: List[BacktestTrade] = []

    for ev in events:
        text = ev.get("text", "")
        author = ev.get("author", "")
        ts = _parse_ts(ev.get("timestamp", ""))
        ents = extract_entities(text)
        label = classify_event(text)
        hi = classify_high_impact_post(author, text)
        sent = score_sentiment(text)
        rumor = is_rumor(text)

        candidates: List[Tuple[str, str]] = []
        for tkr in ents.tickers:
            candidates.append((tkr, "stock"))
        for c in ents.crypto:
            candidates.append((c, "crypto"))

        for symbol, asset_class in candidates:
            px0 = _price_at_or_before(prices, symbol, ts)
            quote: Optional[Quote] = None
            if px0 is not None:
                quote = Quote(symbol=symbol, price=px0, prev_close=px0,
                              volume=1_000_000, avg_volume=1_000_000,
                              daily_atr_pct=0.02, source="backtest")
            inp = ScoringInputs(
                symbol=symbol,
                asset_class=asset_class,
                sentiment=sent,
                source_credibility=settings.source_credibility.get(
                    ev.get("source", ""), 0.5
                ),
                event_timestamp=ts,
                event_label=label,
                high_impact=hi if hi.is_high_impact else None,
                quote=quote,
                duplicate_confirmations=0,
                is_in_universe=symbol in settings.stock_watchlist
                or symbol in settings.crypto_watchlist,
                is_blacklisted=symbol in settings.blacklist,
                is_rumor=rumor,
            )
            scored = score_opportunity(inp)
            if scored.direction != "buy":
                continue
            entry_pair = _next_price(prices, symbol, ts)
            if entry_pair is None:
                continue
            entry_time, entry_px = entry_pair
            series = prices.get(symbol, [])
            after = [(t, p) for (t, p) in series if t > entry_time]
            if not after:
                continue
            future = after[:horizon_bars]
            exit_time = future[-1][0]
            exit_px = future[-1][1]
            reason = "horizon"
            for t, p in future:
                if p <= entry_px * (1 - stop):
                    exit_time, exit_px, reason = t, p, "stop_loss"
                    break
                if p >= entry_px * (1 + take):
                    exit_time, exit_px, reason = t, p, "take_profit"
                    break
            pnl_pct = (exit_px - entry_px) / entry_px
            trades.append(
                BacktestTrade(
                    symbol=symbol,
                    entry_time=entry_time,
                    exit_time=exit_time,
                    entry=entry_px,
                    exit=exit_px,
                    pnl_pct=pnl_pct,
                    reason=reason,
                )
            )

    return _summarize(trades)


def _summarize(trades: List[BacktestTrade]) -> BacktestResult:
    if not trades:
        return BacktestResult(
            n_trades=0, win_rate=0.0, avg_return=0.0, max_drawdown=0.0,
            sharpe=0.0, best_trade=None, worst_trade=None, trades=[],
        )
    rets = [t.pnl_pct for t in trades if t.pnl_pct is not None]
    wins = [r for r in rets if r > 0]
    win_rate = len(wins) / len(rets)
    avg = sum(rets) / len(rets)
    # max drawdown over equity curve
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in rets:
        equity *= (1 + r)
        peak = max(peak, equity)
        dd = (equity - peak) / peak
        max_dd = min(max_dd, dd)
    # Sharpe-like
    if len(rets) > 1:
        mean = avg
        var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
        std = math.sqrt(var)
        sharpe = mean / std if std > 0 else 0.0
    else:
        sharpe = 0.0
    best = max(trades, key=lambda t: t.pnl_pct or -1.0)
    worst = min(trades, key=lambda t: t.pnl_pct or 1.0)
    return BacktestResult(
        n_trades=len(trades),
        win_rate=round(win_rate, 4),
        avg_return=round(avg, 4),
        max_drawdown=round(max_dd, 4),
        sharpe=round(sharpe, 4),
        best_trade=best,
        worst_trade=worst,
        trades=trades,
    )
