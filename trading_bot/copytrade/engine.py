"""Replay a trader's posts as paper trades that follow his rules exactly.

Live account (``live_since`` set): only posts after that moment are acted on,
at the moment the bot actually saw them. A buy seen more than
``max_open_delay_minutes`` after he posted is skipped (the bot was off);
sells and stop moves on positions we hold still happen when seen. Without
``live_since`` every stored post is replayed at its own time (a test on his
old posts).

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

Liquidation uses each market's own maintenance margin (half the initial
margin at that market's max leverage), and his leverage is capped at the
market max. A post can hold several instructions (two trades, or a close plus
a new trade). Candles are only applied once they have ended, so a stop move he
posts inside an hour comes before that hour's low. His trade cards pick which
position a sell means, and on old posts give his own entry and exit prices.

Paper only: nothing here can place a real order.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple

from ..analysis.support_resistance import Bar
from .images import Card, cards as image_cards, chart_symbol
from .parser import Signal, image_symbols, parse_signals


def _rescale(level: Optional[float], price: Optional[float]) -> Optional[float]:
    """His level in the market's units: '0,00517' for kPEPE at 0.00000517, 'SL 80.1' for BTC at 80,660."""
    if not level or not price:
        return level
    while level / price > 300:
        level /= 1000
    while level / price < 1 / 300:
        level *= 1000
    return level


def _fits(trade: "CopyTrade", card: Card) -> bool:
    """Is this his card of the position we copied (same coin, side and his leverage)?"""
    lev = trade.signal.leverage
    return trade.symbol == card.symbol and trade.side == card.side and \
        (card.leverage is None or lev is None or abs(lev - card.leverage) < 0.01)


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
    live_since: Optional[datetime] = None        # None = replay every post (history test)
    max_open_delay_minutes: float = 2.0
    # Default size x this when his post sounds very risky / risky / strong ("klink support").
    size_very_risky: float = 0.25
    size_risky: float = 0.5
    size_strong: float = 1.5
    # True: trades share one account and need free margin (live). False: every trade
    # is sized from the start amount on its own, so a position whose exit was never
    # seen can't block his later calls (history test).
    share_margin: bool = True

    @classmethod
    def from_settings(cls, settings) -> "CopyConfig":
        return cls(
            start_equity=settings.copy_start_equity,
            default_margin_pct=settings.copy_default_margin_pct,
            max_slippage_pct=settings.copy_max_slippage_pct,
            entry_timeout_hours=settings.copy_entry_timeout_hours,
            max_open_delay_minutes=settings.copy_max_open_delay_minutes,
            size_very_risky=settings.copy_size_very_risky,
            size_risky=settings.copy_size_risky,
            size_strong=settings.copy_size_strong,
        )

    def confidence_scale(self, confidence: str) -> float:
        return {"very_risky": self.size_very_risky, "risky": self.size_risky,
                "strong": self.size_strong}.get(confidence, 1.0)


@dataclass
class Post:
    """One message from the trader."""

    message_id: int
    time: datetime
    text: str
    reply_to: Optional[int] = None
    received_at: Optional[datetime] = None     # when the bot saw it
    image_text: Optional[str] = None           # text read from an attached image
    sell_only: bool = False                    # from a discussion topic: never buy from it


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
MaxLeverage = Callable[[str], Optional[float]]
Snapshot = Callable[[int, str], Optional[float]]     # (message id, symbol) -> price when seen


class CopyEngine:
    """Event-by-event paper replay of the trader's posts."""

    def __init__(self, candles: CandleSource, config: Optional[CopyConfig] = None,
                 max_leverage: Optional[MaxLeverage] = None, snapshot: Optional[Snapshot] = None) -> None:
        self.get_candles = candles
        self.max_leverage = max_leverage
        self.snapshot = snapshot
        self.cfg = config or CopyConfig()
        self.cash = self.cfg.start_equity
        self.trades: List[CopyTrade] = []
        self.by_message: Dict[int, List[CopyTrade]] = {}   # post -> trades it opened or updated
        self.posts_by_id: Dict[int, Post] = {}             # every post, to follow reply chains
        self.events: List[Tuple[datetime, float]] = []
        self.outcomes: List[PostOutcome] = []
        self._bars: Dict[str, List[Bar]] = {}
        self._stamps: Dict[str, List[datetime]] = {}
        self._interval: Dict[str, timedelta] = {}
        self._cursor: Dict[int, int] = {}

    # --- prices -------------------------------------------------------------

    def _load(self, symbol: str, since: datetime) -> bool:
        if symbol not in self._bars:
            bars, _ = self.get_candles(symbol, since - timedelta(hours=1))
            self._bars[symbol] = bars
            self._stamps[symbol] = [b.timestamp for b in bars]
            steps = [b.timestamp - a.timestamp for a, b in zip(bars[:50], bars[1:51]) if b.timestamp > a.timestamp]
            self._interval[symbol] = min(steps) if steps else timedelta(minutes=1)
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
        equity = self.cash if self.cfg.share_margin else self.cfg.start_equity
        lev = trade.leverage
        if sig.risk_pct and trade.stop:
            stop_dist = abs(price - trade.stop) / price
            notional = sig.risk_pct * equity / stop_dist
            how = f"risk {sig.risk_pct:.2%}"
        elif sig.size_pct:
            notional = sig.size_pct * equity * lev
            how = f"size {sig.size_pct:.2%}"
        else:  # no size in his post: his tone decides (his own risk/size % already does)
            notional = self.cfg.default_margin_pct * equity * lev
            how = f"default {self.cfg.default_margin_pct:.0%} margin"
            tone = self.cfg.confidence_scale(sig.confidence)
            if tone != 1.0:
                notional *= tone
                how += f" x {tone:.0%} ({sig.confidence.replace('_', ' ')})"
        if sig.size_scale != 1.0:
            notional *= sig.size_scale
            how += f" x {sig.size_scale:.0%}"
        used = sum(o.margin * o.remaining for o in self.trades if o.status == "open")
        free = max(equity - used, 0.0) if self.cfg.share_margin else equity
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
        bars, step = self._bars[trade.symbol], self._interval.get(trade.symbol, timedelta(minutes=1))
        i = self._cursor.get(trade.number, 0)
        # Only candles that ended by ``until``: a candle still forming may make its high/low
        # after a post at ``until`` (e.g. his stop move at 13:17 inside the 13:00 hour).
        while i < len(bars) and bars[i].timestamp + step <= until and trade.status in ("pending", "open"):
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

    def _link(self, message_id: int, trade: CopyTrade) -> None:
        linked = self.by_message.setdefault(message_id, [])
        if trade not in linked:
            linked.append(trade)

    def _linked_trades(self, post: Post) -> Optional[List[CopyTrade]]:
        """Trades behind the post he replied to, following replies to replies (and to images)."""
        rid, seen = post.reply_to, set()
        while rid and rid not in seen and len(seen) < 8:
            seen.add(rid)
            if rid in self.by_message:
                return self.by_message[rid]
            parent = self.posts_by_id.get(rid)
            rid = parent.reply_to if parent else None
        return None

    def _targets_for(self, post: Post, sig: Signal, statuses: Tuple[str, ...]) -> Tuple[List[CopyTrade], str]:
        matches = [t for t in self.trades if t.status in statuses]
        cards = image_cards(post.image_text)
        replied = self._linked_trades(post)
        if replied is not None:
            linked = [t for t in replied if t.status in statuses]
            if sig.symbol:
                linked = [t for t in linked if t.symbol == sig.symbol]
            if linked or not sig.symbol:
                return self._by_leverage(linked, sig), "reply"
        if sig.symbol:
            named = [t for t in matches if t.symbol == sig.symbol]
            mine = [c for c in cards if c.symbol == sig.symbol]
            if mine:   # his card says which of the positions in that coin he sold
                named = [t for t in named if any(_fits(t, c) for c in mine)]
            return self._by_leverage(named, sig), sig.symbol
        if cards:
            hits = [t for t in matches if any(_fits(t, c) for c in cards)]
            return (self._by_leverage(hits, sig), "image") if hits else \
                ([], "card: " + ", ".join(c.describe() for c in cards))
        chart = None if post.sell_only else chart_symbol(post.image_text)
        if chart:
            return [t for t in matches if t.symbol == chart], f"chart: {chart}"
        if post.sell_only:
            return [], "unclear"        # never guess from a discussion topic
        named = image_symbols(post.image_text, {t.symbol for t in matches}) if post.image_text else []
        if len(named) > 1:
            return [], "several: " + ", ".join(named)
        return matches[-1:], "latest"

    @staticmethod
    def _by_leverage(trades: List[CopyTrade], sig: Signal) -> List[CopyTrade]:
        """'Sold my 100x, keeping the 40x': only his position with that leverage."""
        if not sig.leverage:
            return trades
        return [t for t in trades if t.signal.leverage is None or abs(t.signal.leverage - sig.leverage) < 0.01]

    def act_time(self, post: Post) -> datetime:
        """When the bot could act: when it saw the post (live), else when it was posted."""
        if self.cfg.live_since and post.received_at and post.received_at > post.time:
            return post.received_at
        return post.time

    def handle(self, post: Post) -> None:
        at = self.act_time(post)
        for trade in self._active():
            self._advance(trade, at)
        late = (at - post.time).total_seconds() / 60
        if not post.text.strip():
            seen = f": {post.image_text[:80]!r}" if post.image_text else ""
            self.outcomes.append(PostOutcome(post, parse_signals("")[0],
                                             f"image, kept so replies to it can be followed{seen}"))
            return
        for sig in parse_signals(post.text):
            if post.sell_only and sig.action in ("open", "cancel"):
                self.outcomes.append(PostOutcome(post, sig, "discussion topic: only his sells and stop "
                                                            "updates count here, not bought"))
                continue
            if sig.action == "open" and late > self.cfg.max_open_delay_minutes:
                outcome = (f"missed: seen {late:.1f} min after he posted (limit "
                           f"{self.cfg.max_open_delay_minutes:g} min), not bought")
            else:
                outcome = self._apply(replace(post, time=at), sig)
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
                self._link(post.message_id, t)
            return f"cancelled {', '.join(f'#{t.number}' for t in pending)}" if pending else "nothing pending to cancel"
        trades, how = self._targets_for(post, sig, ("open",))
        if not trades:
            pending, _ = self._targets_for(post, sig, ("pending",))
            if sig.action == "close" and pending:
                for t in pending:
                    t.status, t.note = "cancelled", "he closed before our entry filled"
                return f"cancelled pending {', '.join(f'#{t.number}' for t in pending)}"
            return self._ignored(post, sig, how)
        done = []
        for t in trades:
            self._link(post.message_id, t)
            if sig.action == "close":
                price = self._seen_price(post, t.symbol) or self._card_price(post, t) or self._price_at(t.symbol, post.time)
                if price is None:
                    continue
                self._book(t, self._time_at(t.symbol, post.time), price, t.remaining * sig.close_fraction, "his close")
                done.append(f"closed {sig.close_fraction:.0%} of #{t.number} {t.symbol} at {price:g}")
            elif sig.action == "move_sl":
                t.stop = t.entry_price if sig.stop_to_entry else _rescale(sig.new_stop, t.entry_price)
                done.append(f"stop on #{t.number} {t.symbol} now {t.stop:g}")
        suffix = {"latest": " (no coin named: latest position)",
                  "image": " (his trade card shows it)",
                  "reply": " (he replied to it)"}.get(how, " (coin from his chart)" if how.startswith("chart") else "")
        return ("; ".join(done) + suffix + self._image_check(post, trades)) if done else "no price data"

    def _image_check(self, post: Post, acted_on: List[CopyTrade]) -> str:
        """What his picture says, so it's visible that it was read (and if it disagrees)."""
        if not post.image_text:
            return ""
        cards = image_cards(post.image_text)
        if cards:
            if acted_on and not any(_fits(t, c) for t in acted_on for c in cards):
                return f" · ⚠ his card shows {', '.join(c.describe() for c in cards)}"
            return f" · card: {', '.join(c.describe() for c in cards)}"
        chart = chart_symbol(post.image_text)
        if chart:
            acted = {t.symbol for t in acted_on}
            return f" · ⚠ his chart shows {chart}" if acted and chart not in acted else f" · chart: {chart}"
        return " · picture read (no trade card or chart in it)"

    def _ignored(self, post: Post, sig: Signal, how: str = "") -> str:
        """Say exactly why a sell/stop update had nothing to act on."""
        def describe(t: CopyTrade) -> str:
            why = t.fills[-1].reason if t.fills else t.note or t.status
            return f"#{t.number} {t.symbol} ({t.status}: {why})"
        if sig.leverage:
            same = [t for t in self.trades if t.status == "open" and (not sig.symbol or t.symbol == sig.symbol)]
            if same:
                levs = ", ".join(sorted({f"{t.signal.leverage or t.leverage:g}x" for t in same}))
                return f"ignored: he sold his {sig.leverage:g}x position; the copied one(s) are {levs}"
        if how.startswith("card: "):
            return f"ignored: his trade card shows {how[6:]}, not a position the bot copied"
        if how.startswith("several: "):
            return f"ignored: his picture names several open positions ({how[9:]}), not guessing"
        if how.startswith("chart: "):
            return f"ignored: his chart shows {how[7:]}, which the bot doesn't hold"
        replied = self._linked_trades(post)
        if replied:
            return f"ignored: he replied to {', '.join(describe(t) for t in replied)}, which is no longer open"
        if sig.symbol:
            earlier = [t for t in self.trades if t.symbol == sig.symbol]
            if earlier:
                return f"ignored: no open {sig.symbol} position ({describe(earlier[-1])})"
            return f"ignored: no {sig.symbol} position was ever copied"
        if post.sell_only:
            return ("ignored: post in a discussion topic doesn't say what was sold "
                    "(no coin, reply or trade card), not guessing")
        return "ignored: no open position, and the post doesn't say what was sold"

    def _card_price(self, post: Post, trade: CopyTrade) -> Optional[float]:
        """Test on old posts: the mark price on his trade card, when it is his card of this position."""
        if self.cfg.live_since:
            return None
        ref = self._price_at(trade.symbol, post.time)
        for card in image_cards(post.image_text):
            mark = _rescale(card.mark, ref)
            if _fits(trade, card) and mark and ref and abs(mark / ref - 1) < 0.15:
                return mark
        return None

    def _seen_price(self, post: Post, symbol: str) -> Optional[float]:
        """Live account: the price captured when the bot saw the post, if any."""
        if self.cfg.live_since and self.snapshot:
            return self.snapshot(post.message_id, symbol)
        return None

    def _time_at(self, symbol: str, t: datetime) -> datetime:
        i = self._index_at(symbol, t)
        return self._bars[symbol][i].timestamp if i is not None else t

    def _resolve_add(self, post: Post, sig: Signal) -> Tuple[Optional[Signal], str]:
        """'Longet mer' (reply) / 'Longer mer TESLA': the coin and his leverage come from that position."""
        linked = self._linked_trades(post) or []
        base = next((t for t in reversed(linked) if sig.symbol in (None, t.symbol)), None)
        if base is None and sig.symbol:
            base = next((t for t in reversed(self.trades) if t.symbol == sig.symbol and t.side == sig.side), None)
        if base is None and sig.symbol is None:
            return None, "ignored: he added to a position but names no coin and didn't reply to one of his trades"
        symbol = sig.symbol or base.symbol
        lev = sig.leverage or (base.signal.leverage if base else None)
        note = f"adds to his {symbol} position" + (f" (#{base.number}, {lev:g}x)" if base and lev else "")
        return replace(sig, symbol=symbol, leverage=lev, notes=sig.notes + [note]), ""

    def _apply_open(self, post: Post, sig: Signal) -> str:
        if sig.add:
            sig, why = self._resolve_add(post, sig)
            if sig is None:
                return why
        card = next((c for c in image_cards(post.image_text) if c.symbol == sig.symbol and c.side == sig.side), None)
        if sig.leverage is None and card and card.leverage:   # "Velger long META" + his 20x card
            sig = replace(sig, leverage=card.leverage, notes=sig.notes + [f"{card.leverage:g}x from his trade card"])
        number = len(self.trades) + 1
        trade = CopyTrade(number, post.message_id, sig.symbol, sig.side, sig, post.time,
                          leverage=sig.leverage or 1.0, mm=self.cfg.maintenance_margin)
        market_max = self.max_leverage(sig.symbol) if self.max_leverage else None
        capped = ""
        if market_max:
            trade.mm = 1 / (2 * market_max)
            if trade.leverage > market_max:
                capped = f" (his {trade.leverage:g}x capped to the market max {market_max:g}x)"
                trade.leverage = market_max
        self.trades.append(trade)
        self._link(post.message_id, trade)
        if not self._load(sig.symbol, post.time):
            trade.status, trade.note = "skipped", f"no price data for {sig.symbol}"
            return f"#{number} skipped: no price data for {sig.symbol}"
        i = self._index_at(sig.symbol, post.time)
        if i is None:
            trade.status, trade.note = "skipped", "post is newer than the price data"
            return f"#{number} skipped: no price after the post yet"
        bars = self._bars[sig.symbol]
        if bars[i].timestamp - post.time > max(timedelta(hours=3), 2 * self._interval[sig.symbol]):
            trade.status, trade.note = "skipped", f"no price from when he posted (prices start {bars[0].timestamp:%Y-%m-%d})"
            return f"#{number} skipped: {trade.note}"
        self._cursor[number] = i
        bar = bars[i]
        price = bar.open
        seen = self._seen_price(post, sig.symbol)
        card_entry = _rescale(card.entry, price) if card and not seen else None
        if seen:
            price, bar = seen, Bar(post.time, seen, seen, seen, seen)   # fill at the live price, now
        elif card_entry and abs(card_entry / price - 1) < 0.15:
            price = card_entry   # test on old posts: his own fill from the card in this post
        trade.stop = trade.initial_stop = _rescale(sig.stop_loss, price)
        trade.targets = [_rescale(tp, price) for tp in sig.take_profits]
        low, high = _rescale(sig.entry_low, price), _rescale(sig.entry_high, price)
        if low is not None:
            trade.his_entry = (low + high) / 2
        d = trade.direction
        if trade.stop and d * (price - trade.stop) <= 0 and (trade.his_entry is None or low is None):
            trade.status = "skipped"
            trade.note = f"price {price:g} already past his stop {trade.stop:g}"
            return f"#{number} not bought: {trade.note}"
        if trade.his_entry is None:
            trade.his_entry = price
            self._open(trade, bar.timestamp, price)
            if trade.status == "skipped":
                return f"#{number} not bought: {trade.note} (all margin is in open positions)"
            return f"#{number} opened at {price:g} (market){self._size_note(trade)}{capped}"
        edge = high if d == 1 else low
        if d * (price - edge) <= edge * self.cfg.max_slippage_pct:
            self._open(trade, bar.timestamp, price)
            if trade.status == "skipped":
                return f"#{number} not bought: {trade.note} (all margin is in open positions)"
            return f"#{number} opened at {price:g} (his entry {self._entry_text(sig)}){self._size_note(trade)}{capped}"
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


def feed_snapshot(feed, snapshots: Dict[int, Dict[str, float]]) -> Snapshot:
    """Look up a stored price snapshot for (message id, symbol) via the feed's market names."""
    def lookup(message_id: int, symbol: str) -> Optional[float]:
        mids = snapshots.get(message_id)
        return feed.price_from_snapshot(mids, symbol) if mids else None
    return lookup


def replay(posts: List[Post], candles: CandleSource, config: Optional[CopyConfig] = None,
           now: Optional[datetime] = None, max_leverage: Optional[MaxLeverage] = None,
           snapshot: Optional[Snapshot] = None) -> CopyResult:
    """Run all posts (any order) through a fresh engine."""
    engine = CopyEngine(candles, config, max_leverage, snapshot)
    engine.posts_by_id = {p.message_id: p for p in posts}
    since = engine.cfg.live_since
    chosen = [p for p in posts if since is None or p.time >= since]
    for post in sorted(chosen, key=lambda p: (engine.act_time(p), p.message_id)):
        engine.handle(post)
    return engine.finish(now)
