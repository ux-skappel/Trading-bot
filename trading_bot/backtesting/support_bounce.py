"""Backtest of the "bounce off support" strategy on daily candles.

A short-term trade: buy the bounce, get out before it falls again, and keep
every loss small. Long only, one position per symbol at a time.

Entry

* Each day, find support/resistance in the previous ``lookback_bars`` candles
  with the same detector the live bot uses. A swing low only counts once
  ``sr_window`` later candles have confirmed it, so there is no look-ahead.
* Signal: the candle's low dips into a support zone (level +/- tolerance)
  with at least ``min_touches`` touches, does not break below the zone, and
  the candle closes green above the level.
* Plan: stop just below that support (if the support breaks, the idea is
  wrong), target just below the nearest resistance (where bounces tend to
  stall), or +``default_take_pct`` if there is none. Skip the signal when
  reward/risk is below ``min_reward_risk``.
* Enter at the next day's open, but only if reward/risk still holds at that
  price (a gap up can ruin it).

Exit — whichever comes first

* Stop or target. When both are inside one candle the stop is assumed to
  come first; a gap through either fills at the open.
* With ``trailing`` on: once price has risen ``breakeven_at_r`` times the
  initial risk, the stop moves up to break-even (fees included), then follows
  ``trail_atr`` x ATR below the highest high since entry. The stop for a day
  is set from the previous days only, never from that day's own range.
* With ``max_hold_days`` set: sell at that day's close.

Position size

* ``risk_per_trade`` set: size the position so that hitting the stop costs
  that fraction of the account (e.g. 1%), using at most ``leverage`` x.
* Otherwise the position is a fixed ``leverage`` x the account.

Costs and liquidation follow Liquid's perps (liquid.trade, which routes to
Hyperliquid). Leverage is for this backtest only; the paper/live bot never
uses it.

* Taker fees (0.095% per side) on the full position on entry and exit, and
  funding (base rate 0.01% per 8 hours) on the full position per day held.
* Maintenance margin is 1.25% of the position (Liquid's figure for 40x
  markets). If price falls to where only that is left before the stop is
  reached, the position is liquidated there and you keep what remains. A gap
  straight through it loses everything.
* Once less than 1% of the starting money is left, the account stops trading.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from ..analysis.support_resistance import (
    Bar,
    SupportResistance,
    average_true_range,
    detect_levels,
    suggest_stop_take,
)

# Below 1% of the starting money (e.g. under $10 of $1,000 — Liquid's minimum
# position) the account can't trade any more and counts as wiped out.
WIPED_OUT = 0.01


@dataclass
class BounceParams:
    """Tunable strategy settings."""

    lookback_bars: int = 126        # ~6 months of daily candles, like the live bot
    sr_window: int = 5
    max_levels: int = 3
    min_touches: int = 2
    min_reward_risk: float = 1.5
    default_take_pct: float = 0.10
    max_stop_pct: float = 0.10
    fee_pct: float = 0.00095                # Liquid perps taker fee, per side, on the full position
    leverage: float = 1.0                   # fixed size, or the cap when risk_per_trade is set
    risk_per_trade: Optional[float] = None  # e.g. 0.01 = a stop-out costs 1% of the account
    max_hold_days: Optional[int] = None     # None = hold until stop or target
    trailing: bool = False
    breakeven_at_r: float = 1.0
    trail_atr: float = 1.0
    funding_pct_per_day: float = 0.0003     # Liquid base funding 0.01%/8h, on the full position
    maintenance_margin: float = 0.0125      # Liquid: half the initial margin at 40x

    @classmethod
    def from_settings(cls, settings, **overrides) -> "BounceParams":
        """Defaults taken from the live bot's settings, with optional overrides."""
        values = dict(
            sr_window=settings.sr_window,
            max_levels=settings.sr_max_levels,
            min_reward_risk=settings.min_reward_risk,
            default_take_pct=settings.default_take_profit_pct,
            max_stop_pct=settings.max_stop_loss_pct,
        )
        values.update(overrides)
        return cls(**values)


@dataclass
class BounceTrade:
    """One completed (or force-closed) trade."""

    symbol: str
    entry_time: datetime
    entry_price: float
    stop: float             # initial stop, just below the support
    target: float
    support: float
    support_touches: int
    position_size: float    # position as a multiple of the account (2.0 = 2x)
    exit_time: datetime
    exit_price: float
    exit_reason: str        # stop | trail | target | time | liquidated | end_of_data
    return_pct: float       # on your own money: after leverage, fees and funding
    days_held: int          # 1 = sold the same day it was bought
    levels: Optional[SupportResistance] = None   # all levels as seen on the signal day


@dataclass
class BounceResult:
    """Trades and equity curves for one symbol."""

    symbol: str
    trades: List[BounceTrade] = field(default_factory=list)
    equity: List[Tuple[datetime, float]] = field(default_factory=list)     # starts at 1.0
    buy_hold: List[Tuple[datetime, float]] = field(default_factory=list)   # 1x, starts at 1.0
    time_in_market: float = 0.0

    @property
    def total_return(self) -> float:
        return self.equity[-1][1] - 1 if self.equity else 0.0

    @property
    def buy_hold_return(self) -> float:
        return self.buy_hold[-1][1] - 1 if self.buy_hold else 0.0

    @property
    def win_rate(self) -> Optional[float]:
        if not self.trades:
            return None
        return sum(1 for t in self.trades if t.return_pct > 0) / len(self.trades)

    @property
    def avg_return(self) -> Optional[float]:
        if not self.trades:
            return None
        return sum(t.return_pct for t in self.trades) / len(self.trades)

    @property
    def max_drawdown(self) -> float:
        peak, worst = 0.0, 0.0
        for _, value in self.equity:
            peak = max(peak, value)
            if peak > 0:
                worst = min(worst, value / peak - 1)
        return worst

    @property
    def liquidations(self) -> int:
        return sum(1 for t in self.trades if t.exit_reason == "liquidated")


@dataclass
class _Signal:
    stop: float
    target: float
    support: float
    touches: int
    atr: float
    levels: SupportResistance


@dataclass
class _Position:
    entry_index: int
    entry_price: float
    size: float
    signal: _Signal
    stop: float             # current stop; starts at signal.stop, only ever moves up
    high: float             # highest high since entry


def _find_signal(bars: List[Bar], t: int, p: BounceParams) -> Optional[_Signal]:
    bar = bars[t]
    if bar.close <= bar.open:
        return None
    window = bars[t - p.lookback_bars + 1:t + 1]
    sr = detect_levels(window, window=p.sr_window, max_levels=p.max_levels)
    tol = sr.tolerance_pct
    touched = [
        s for s in sr.supports
        if s.touches >= p.min_touches
        and s.price * (1 - tol) <= bar.low <= s.price * (1 + tol)
        and bar.close > s.price
    ]
    if not touched:
        return None
    support = max(touched, key=lambda s: s.price)
    plan = suggest_stop_take(
        replace(sr, supports=[support]),
        entry=bar.close,
        default_stop_pct=p.max_stop_pct,
        default_take_pct=p.default_take_pct,
        max_stop_pct=p.max_stop_pct,
    )
    if plan.stop_level is None or plan.reward_risk is None or plan.reward_risk < p.min_reward_risk:
        return None
    return _Signal(plan.stop_loss, plan.take_profit, support.price, support.touches,
                   average_true_range(window), sr)


def _position_size(p: BounceParams, entry: float, stop: float) -> float:
    if p.risk_per_trade is None:
        return p.leverage
    return min(p.leverage, p.risk_per_trade / ((entry - stop) / entry))


def backtest_support_bounce(
    bars: List[Bar], symbol: str = "", params: Optional[BounceParams] = None
) -> BounceResult:
    """Run the support-bounce strategy over daily bars (oldest first)."""
    p = params or BounceParams()
    start = p.lookback_bars - 1
    if len(bars) < start + 2:
        return BounceResult(symbol)

    fee = p.fee_pct
    cash = 1.0
    trades: List[BounceTrade] = []
    equity: List[Tuple[datetime, float]] = []
    position: Optional[_Position] = None
    pending: Optional[_Signal] = None
    bars_in_market = 0

    def close_position(exit_index: int, price: float, reason: str) -> None:
        nonlocal cash, position
        assert position is not None
        size = position.size
        days = exit_index - position.entry_index + 1
        move = price / position.entry_price - 1
        costs = size * (fee * (2 + move) + p.funding_pct_per_day * days)
        ret = max(-1.0, size * move - costs)
        cash *= 1 + ret
        s = position.signal
        trades.append(BounceTrade(
            symbol=symbol,
            entry_time=bars[position.entry_index].timestamp,
            entry_price=position.entry_price,
            stop=s.stop, target=s.target, support=s.support, support_touches=s.touches,
            position_size=size,
            exit_time=bars[exit_index].timestamp,
            exit_price=price,
            exit_reason=reason,
            return_pct=ret,
            days_held=days,
            levels=s.levels,
        ))
        position = None

    for t in range(start, len(bars)):
        bar = bars[t]

        if pending is not None:
            risk, reward = bar.open - pending.stop, pending.target - bar.open
            if risk > 0 and reward > 0 and reward >= p.min_reward_risk * risk:
                size = _position_size(p, bar.open, pending.stop)
                position = _Position(t, bar.open, size, pending, pending.stop, bar.open)
            pending = None

        if position is not None:
            s, stop = position.signal, position.stop
            stop_reason = "stop" if stop <= s.stop else "trail"
            # Price drop that leaves only the maintenance margin; <=1x can't be liquidated.
            liq = (position.entry_price * (1 - (1 / position.size - p.maintenance_margin))
                   if position.size > 1 else 0.0)
            if bar.open <= liq:
                close_position(t, bar.open, "liquidated")
            elif bar.open <= stop:
                close_position(t, bar.open, stop_reason)
            elif bar.open >= s.target:
                close_position(t, bar.open, "target")
            elif bar.low <= max(stop, liq):
                # Falling price reaches whichever level is higher first.
                if stop >= liq:
                    close_position(t, stop, stop_reason)
                else:
                    close_position(t, liq, "liquidated")
            elif bar.high >= s.target:
                close_position(t, s.target, "target")
            elif p.max_hold_days and t - position.entry_index + 1 >= p.max_hold_days:
                close_position(t, bar.close, "time")

        if position is not None:
            if p.trailing:
                position.high = max(position.high, bar.high)
                initial_risk = position.entry_price - position.signal.stop
                if position.high >= position.entry_price + p.breakeven_at_r * initial_risk:
                    position.stop = max(
                        position.stop,
                        position.entry_price * (1 + 2 * fee),
                        position.high - p.trail_atr * position.signal.atr,
                    )
            bars_in_market += 1
            move = bar.close / position.entry_price - 1
            equity.append((bar.timestamp, cash * max(0.0, 1 + position.size * (move - fee))))
        else:
            equity.append((bar.timestamp, cash))
            if cash >= WIPED_OUT and t < len(bars) - 1:
                pending = _find_signal(bars, t, p)

    if position is not None:
        close_position(len(bars) - 1, bars[-1].close, "end_of_data")
        equity[-1] = (bars[-1].timestamp, cash)

    base = bars[start].close * (1 + fee)
    buy_hold = [(b.timestamp, b.close * (1 - fee) / base) for b in bars[start:]]
    return BounceResult(
        symbol=symbol,
        trades=trades,
        equity=equity,
        buy_hold=buy_hold,
        time_in_market=bars_in_market / len(equity),
    )


def shared_account(results: List[BounceResult]) -> Tuple[float, float]:
    """(total return, max drawdown) of ONE account taking every trade of every symbol.

    Each trade is sized on the whole account (e.g. risking 2% of it), in the
    order trades close, so this is what following all signals would have done.
    Per-symbol results instead give each symbol its own account and hide this.
    """
    equity, peak, worst = 1.0, 1.0, 0.0
    for t in sorted((t for r in results for t in r.trades), key=lambda t: t.exit_time):
        equity *= 1 + t.return_pct
        peak = max(peak, equity)
        worst = min(worst, equity / peak - 1)
    return equity - 1, worst


def summarize(results: List[BounceResult]) -> Dict[str, Optional[float]]:
    """Pool all trades across symbols into headline numbers."""
    trades = [t for r in results for t in r.trades]
    returns = [t.return_pct for t in trades]
    gains = sum(r for r in returns if r > 0)
    losses = -sum(r for r in returns if r < 0)
    account, account_dd = shared_account(results)
    return {
        "account_return": account,
        "account_max_drawdown": account_dd,
        "symbols": len(results),
        "trades": len(trades),
        "win_rate": sum(1 for r in returns if r > 0) / len(returns) if returns else None,
        "avg_return": sum(returns) / len(returns) if returns else None,
        "worst_trade": min(returns) if returns else None,
        "profit_factor": gains / losses if losses else None,
        "beat_buy_hold": sum(1 for r in results if r.total_return > r.buy_hold_return),
        "liquidations": sum(r.liquidations for r in results),
        "wiped_out": sum(1 for r in results if r.equity and r.equity[-1][1] < WIPED_OUT),
        "avg_days_held": sum(t.days_held for t in trades) / len(trades) if trades else None,
    }
