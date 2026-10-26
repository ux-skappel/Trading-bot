"""Replay a trader's posts as paper trades that follow his rules exactly.

Entry: if the price when the post arrives is still within ``max_slippage_pct``
of his entry (or better), fill right away; otherwise wait at his entry price
for ``entry_timeout_hours``. Posts without an entry are filled at once.

Size: his risk % (with his stop) if he gives one, else his size % of the
account, else ``default_margin_pct`` of the account as margin — always at his
leverage (1x if he names none).

Exits: his stop loss, his targets (the position is split equally across
them), and his follow-up posts (close / close part / move stop / cancel).
Costs follow Liquid perps: taker fee per side, base funding per day (longs
pay, shorts receive), and isolated-margin liquidation at the maintenance
margin. When a candle touches both a stop and a target, the stop is assumed
to come first.

Paper only: nothing here can place a real order.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple

from ..analysis.support_resistance import Bar
from .parser import Signal, parse_signal


@dataclass
class CopyConfig:
    """Paper-copy settings (sizes are fractions of the account)."""

    start_equity: float = 10_000.0
    default_margin_pct: float = 0.10
    max_slippage_pct: float = 0.01
    entry_timeout_hours: float = 24.0
    fee_pct: float = 0.00095
    funding_pct_per_day: float = 0.0003
    maintenance_margin: float = 0.0125

    @classmethod
    def from_settings(cls, settings) -> "CopyConfig":
        return cls(
            start_equity=settings.copy_start_equity,
            default_margin_pct=settings.copy_default_margin_pct,
            max_slippage_pct=settings.copy_max_slippage_pct,
            entry_timeout_hours=settings.copy_entry_timeout_hours,
        )


@dataclass
class Post:
    """One message from the trader."""

    message_id: int
    time: datetime
    text: str
    reply_to: Optional[int] = None


@dataclass
class Fill:
    time: datetime
    price: float
    fraction: float          # share of the original position
    reason: str              # tp1, tp2, ..., stop, his close, liquidated
    pnl: float


@dataclass
class CopyTrade:
    """A paper copy of one of his trade posts."""

    number: int
    message_id: int
    symbol: str
    side: str
    signal: Signal
    posted_at: datetime
    leverage: float
    status: str = "pending"          # pending | open | closed | cancelled | expired | skipped
    his_entry: Optional[float] = None
    limit_price: Optional[float] = None
    entry_time: Optional[datetime] = None
    entry_price: Optional[float] = None
    notional: float = 0.0
    margin: float = 0.0
    initial_stop: Optional[float] = None
    stop: Optional[float] = None
    targets: List[float] = field(default_factory=list)
    targets_hit: int = 0
    remaining: float = 1.0
    fills: List[Fill] = field(default_factory=list)
    pnl: float = 0.0
    unrealized: float = 0.0
    note: str = ""
    mm: float = CopyConfig.maintenance_margin

    @property
    def direction(self) -> int:
        return 1 if self.side == "long" else -1

    @property
    def liquidation_price(self) -> Optional[float]:
        if self.leverage <= 1 or self.entry_price is None:
            return None
        return self.entry_price * (1 - self.direction * (1 / self.leverage - self.mm))

    @property
    def return_on_margin(self) -> Optional[float]:
        return (self.pnl + self.unrealized) / self.margin if self.margin else None

    @property
    def delay_cost(self) -> Optional[float]:
        """How much worse our entry was than his stated entry (positive = worse)."""
        if self.his_entry is None or self.entry_price is None:
            return None
        return self.direction * (self.entry_price / self.his_entry - 1)

    @property
    def exit_time(self) -> Optional[datetime]:
        return self.fills[-1].time if self.status == "closed" and self.fills else None


@dataclass
class PostOutcome:
    post: Post
    signal: Signal
    outcome: str


@dataclass
class CopyResult:
    trades: List[CopyTrade]
    posts: List[PostOutcome]
    equity: List[Tuple[datetime, float]]       # realized equity over time
    start_equity: float

    @property
    def closed(self) -> List[CopyTrade]:
        return [t for t in self.trades if t.status == "closed"]

    @property
    def realized_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    @property
    def total_return(self) -> float:
        return self.realized_pnl / self.start_equity

    @property
    def win_rate(self) -> Optional[float]:
        closed = self.closed
        return sum(1 for t in closed if t.pnl > 0) / len(closed) if closed else None

    @property
    def max_drawdown(self) -> float:
        peak, worst = self.start_equity, 0.0
        for _, value in self.equity:
            peak = max(peak, value)
            worst = min(worst, value / peak - 1)
        return worst

    @property
    def avg_delay_cost(self) -> Optional[float]:
        costs = [t.delay_cost for t in self.trades if t.delay_cost is not None]
        return sum(costs) / len(costs) if costs else None


CandleSource = Callable[[str, datetime], Tuple[List[Bar], str]]


class CopyEngine:
    """Event-by-event paper replay of the trader's posts."""

    def __init__(self, candles: CandleSource, config: Optional[CopyConfig] = None) -> None:
        self.get_candles = candles
        self.cfg = config or CopyConfig()
        self.cash = self.cfg.start_equity
        self.trades: List[CopyTrade] = []
        self.by_message: Dict[int, CopyTrade] = {}
        self.events: List[Tuple[datetime, float]] = []
        self.outcomes: List[PostOutcome] = []
        self._bars: Dict[str, List[Bar]] = {}
        self._stamps: Dict[str, List[datetime]] = {}
        self._cursor: Dict[int, int] = {}

    # --- prices -------------------------------------------------------------

    def _load(self, symbol: str, since: datetime) -> bool:
        if symbol not in self._bars:
            bars, _ = self.get_candles(symbol, since - timedelta(hours=1))
            self._bars[symbol] = bars
            self._stamps[symbol] = [b.timestamp for b in bars]
        return bool(self._bars[symbol])

    def _index_at(self, symbol: str, t: datetime) -> Optional[int]:
        """Index of the first candle starting at or after ``t`` (next tradable price)."""
        i = bisect.bisect_left(self._stamps[symbol], t)
        return i if i < len(self._bars[symbol]) else None

    def _price_at(self, symbol: str, t: datetime) -> Optional[float]:
        i = self._index_at(symbol, t)
        if i is not None:
            return self._bars[symbol][i].open
        bars = self._bars.get(symbol) or []
        return bars[-1].close if bars and bars[-1].timestamp <= t else None

    # --- money --------------------------------------------------------------

    def _book(self, trade: CopyTrade, t: datetime, price: float, fraction: float, reason: str) -> None:
        fraction = min(fraction, trade.remaining)
        if fraction <= 1e-12:
            return
        size = trade.notional * fraction
        days = max((t - trade.entry_time).total_seconds() / 86400, 0.0)
        move = trade.direction * (price / trade.entry_price - 1)
        pnl = size * move - size * (price / trade.entry_price) * self.cfg.fee_pct \
            - trade.direction * size * self.cfg.funding_pct_per_day * days
        if reason == "liquidated":
            pnl = max(pnl, -trade.margin * fraction)
        trade.remaining -= fraction
        trade.pnl += pnl
        self.cash += pnl
        trade.fills.append(Fill(t, price, fraction, reason, pnl))
        self.events.append((t, pnl))
        if trade.remaining <= 1e-9:
            trade.remaining = 0.0
            trade.status = "closed"

    def _open(self, trade: CopyTrade, t: datetime, price: float) -> None:
        sig = trade.signal
        equity = self.cash
        lev = trade.leverage
        if sig.risk_pct and trade.stop:
            stop_dist = abs(price - trade.stop) / price
            notional = sig.risk_pct * equity / stop_dist
            how = f"risk {sig.risk_pct:.2%}"
        elif sig.size_pct:
            notional = sig.size_pct * equity * lev
            how = f"size {sig.size_pct:.2%}"
        else:
            notional = self.cfg.default_margin_pct * equity * lev
            how = f"default {self.cfg.default_margin_pct:.0%} margin"
        used = sum(o.margin * o.remaining for o in self.trades if o.status == "open")
        free = max(equity - used, 0.0)
        if notional / lev > free:
            notional = free * lev
            how += ", cut to free margin"
        if notional <= 0:
            trade.status = "skipped"
            trade.note = "no free margin"
            return
        trade.notional, trade.margin = notional, notional / lev
        trade.entry_time, trade.entry_price = t, price
        trade.status = "open"
        fee = notional * self.cfg.fee_pct
        trade.pnl -= fee
        self.cash -= fee
        self.events.append((t, -fee))
        trade.note = f"{how}, {lev:g}x, ${notional:,.0f} position"

    # --- walking candles ------------------------------------------------------

    def _advance(self, trade: CopyTrade, until: datetime) -> None:
        bars = self._bars[trade.symbol]
        i = self._cursor.get(trade.number, 0)
        while i < len(bars) and bars[i].timestamp < until and trade.status in ("pending", "open"):
            self._step(trade, bars[i])
            i += 1
        self._cursor[trade.number] = i

    def _step(self, trade: CopyTrade, bar: Bar) -> None:
        d = trade.direction
        filled_now = False
        if trade.status == "pending":
            if bar.timestamp - trade.posted_at > timedelta(hours=self.cfg.entry_timeout_hours):
                trade.status = "expired"
                trade.note = f"entry {trade.limit_price:g} not reached in {self.cfg.entry_timeout_hours:g}h"
                return
            lp = trade.limit_price
            if (d == 1 and bar.low <= lp) or (d == -1 and bar.high >= lp):
                price = min(bar.open, lp) if d == 1 else max(bar.open, lp)
                self._open(trade, bar.timestamp, price)
                filled_now = True
            if trade.status != "open":
                return

        liq = trade.liquidation_price
        stop = trade.stop
        t = bar.timestamp

        def beyond(level: Optional[float], value: float) -> bool:
            return level is not None and (value <= level if d == 1 else value >= level)

        part = 1 / len(trade.targets) if trade.targets else 0
        # On the candle a waiting entry fills, its open came before the fill and
        # its best price may have too: only the stop/liquidation is checked there.
        if not filled_now:
            if beyond(liq, bar.open):
                self._book(trade, t, bar.open, trade.remaining, "liquidated")
                return
            if beyond(stop, bar.open):
                self._book(trade, t, bar.open, trade.remaining, self._stop_reason(trade))
                return
            while trade.targets_hit < len(trade.targets) and \
                    (bar.open >= trade.targets[trade.targets_hit] if d == 1
                     else bar.open <= trade.targets[trade.targets_hit]):
                trade.targets_hit += 1
                last = trade.targets_hit == len(trade.targets)
                self._book(trade, t, bar.open, trade.remaining if last else part, f"tp{trade.targets_hit}")
            if trade.status != "open":
                return
        worst = bar.low if d == 1 else bar.high
        levels = [lv for lv in (stop, liq) if lv is not None]
        if levels:
            first = max(levels) if d == 1 else min(levels)
            if beyond(first, worst):
                reason = "liquidated" if first == liq and (stop is None or first != stop) else self._stop_reason(trade)
                self._book(trade, t, first, trade.remaining, reason)
                return
        best = bar.high if d == 1 else bar.low
        while not filled_now and trade.targets_hit < len(trade.targets) and \
                (best >= trade.targets[trade.targets_hit] if d == 1 else best <= trade.targets[trade.targets_hit]):
            price = trade.targets[trade.targets_hit]
            trade.targets_hit += 1
            last = trade.targets_hit == len(trade.targets)
            self._book(trade, t, price, trade.remaining if last else part, f"tp{trade.targets_hit}")

    @staticmethod
    def _stop_reason(trade: CopyTrade) -> str:
        return "stop" if trade.stop == trade.initial_stop else "moved stop"

    # --- posts ----------------------------------------------------------------

    def _active(self) -> List[CopyTrade]:
        return [t for t in self.trades if t.status in ("pending", "open")]

    def _targets_for(self, post: Post, sig: Signal, statuses: Tuple[str, ...]) -> Tuple[List[CopyTrade], str]:
        if post.reply_to and post.reply_to in self.by_message:
            trade = self.by_message[post.reply_to]
            return ([trade] if trade.status in statuses else []), "reply"
        matches = [t for t in self.trades if t.status in statuses]
        if sig.symbol:
            return [t for t in matches if t.symbol == sig.symbol], sig.symbol
        return matches[-1:], "latest"

    def handle(self, post: Post) -> None:
        for trade in self._active():
            self._advance(trade, post.time)
        sig = parse_signal(post.text)
        outcome = self._apply(post, sig)
        self.outcomes.append(PostOutcome(post, sig, outcome))

    def _apply(self, post: Post, sig: Signal) -> str:
        if sig.action == "info":
            return "not a trade instruction: " + "; ".join(sig.notes)
        if sig.action == "open":
            return self._apply_open(post, sig)
        if sig.action == "cancel":
            pending, how = self._targets_for(post, sig, ("pending",))
            for t in pending:
                t.status, t.note = "cancelled", "cancelled by his post"
                self.by_message[post.message_id] = t
            return f"cancelled {', '.join(f'#{t.number}' for t in pending)}" if pending else "nothing pending to cancel"
        trades, how = self._targets_for(post, sig, ("open",))
        if not trades:
            pending, _ = self._targets_for(post, sig, ("pending",))
            if sig.action == "close" and pending:
                for t in pending:
                    t.status, t.note = "cancelled", "he closed before our entry filled"
                return f"cancelled pending {', '.join(f'#{t.number}' for t in pending)}"
            done_before = [t for t in self.trades if t.status == "closed"
                           and (not sig.symbol or t.symbol == sig.symbol)]
            if done_before:
                last = done_before[-1]
                return f"ignored: #{last.number} {last.symbol} was already closed ({last.fills[-1].reason})"
            return f"ignored: no open position{' for ' + sig.symbol if sig.symbol else ''}"
        done = []
        for t in trades:
            self.by_message[post.message_id] = t
            if sig.action == "close":
                price = self._price_at(t.symbol, post.time)
                if price is None:
                    continue
                self._book(t, self._time_at(t.symbol, post.time), price, t.remaining * sig.close_fraction, "his close")
                done.append(f"closed {sig.close_fraction:.0%} of #{t.number} at {price:g}")
            elif sig.action == "move_sl":
                t.stop = t.entry_price if sig.stop_to_entry else sig.new_stop
                done.append(f"stop on #{t.number} now {t.stop:g}")
        suffix = " (no coin named: latest position)" if how == "latest" else ""
        return "; ".join(done) + suffix if done else "no price data"

    def _time_at(self, symbol: str, t: datetime) -> datetime:
        i = self._index_at(symbol, t)
        return self._bars[symbol][i].timestamp if i is not None else t

    def _apply_open(self, post: Post, sig: Signal) -> str:
        number = len(self.trades) + 1
        trade = CopyTrade(number, post.message_id, sig.symbol, sig.side, sig, post.time,
                          leverage=sig.leverage or 1.0, mm=self.cfg.maintenance_margin)
        trade.stop = trade.initial_stop = sig.stop_loss
        trade.targets = list(sig.take_profits)
        if sig.entry_low is not None:
            trade.his_entry = (sig.entry_low + sig.entry_high) / 2
        self.trades.append(trade)
        self.by_message[post.message_id] = trade
        if not self._load(sig.symbol, post.time):
            trade.status, trade.note = "skipped", f"no price data for {sig.symbol}"
            return f"#{number} skipped: no price data for {sig.symbol}"
        i = self._index_at(sig.symbol, post.time)
        if i is None:
            trade.status, trade.note = "skipped", "post is newer than the price data"
            return f"#{number} skipped: no price after the post yet"
        self._cursor[number] = i
        bar = self._bars[sig.symbol][i]
        price = bar.open
        if trade.his_entry is None:
            trade.his_entry = price
            self._open(trade, bar.timestamp, price)
            return f"#{number} opened at {price:g} (market){self._size_note(trade)}"
        d = trade.direction
        edge = sig.entry_high if d == 1 else sig.entry_low
        if d * (price - edge) <= edge * self.cfg.max_slippage_pct:
            self._open(trade, bar.timestamp, price)
            return f"#{number} opened at {price:g} (his entry {self._entry_text(sig)}){self._size_note(trade)}"
        trade.limit_price = edge
        trade.status = "pending"
        return (f"#{number} waiting for {edge:g}: price {price:g} ran more than "
                f"{self.cfg.max_slippage_pct:.0%} past his entry")

    @staticmethod
    def _entry_text(sig: Signal) -> str:
        return f"{sig.entry_low:g}" if sig.entry_low == sig.entry_high else f"{sig.entry_low:g}-{sig.entry_high:g}"

    @staticmethod
    def _size_note(trade: CopyTrade) -> str:
        return f" · {trade.note}" if trade.note else ""

    def finish(self, now: Optional[datetime] = None) -> CopyResult:
        now = now or datetime.utcnow()
        for trade in self._active():
            self._advance(trade, now)
        for trade in self._active():
            if trade.status == "open":
                last = self._bars[trade.symbol][-1].close
                size = trade.notional * trade.remaining
                trade.unrealized = size * trade.direction * (last / trade.entry_price - 1)
        equity, running = [], self.cfg.start_equity
        for t, delta in sorted(self.events, key=lambda e: e[0]):
            running += delta
            equity.append((t, running))
        return CopyResult(self.trades, self.outcomes, equity, self.cfg.start_equity)


def replay(posts: List[Post], candles: CandleSource, config: Optional[CopyConfig] = None,
           now: Optional[datetime] = None) -> CopyResult:
    """Run all posts (any order) through a fresh engine."""
    engine = CopyEngine(candles, config)
    for post in sorted(posts, key=lambda p: (p.time, p.message_id)):
        engine.handle(post)
    return engine.finish(now)
