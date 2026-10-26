"""Backtest of the "bounce off support" strategy on daily candles.

Rules (long only, one position per symbol at a time):

* Each day, find support/resistance in the previous ``lookback_bars`` candles
  with the same detector the live bot uses. A swing low only counts once
  ``sr_window`` later candles have confirmed it, so there is no look-ahead.
* Signal: the candle's low dips into a support zone (level +/- tolerance)
  with at least ``min_touches`` touches, does not break below the zone, and
  the candle closes green above the level.
* Plan: stop just below that support, target just below the nearest
  resistance (+``default_take_pct`` if there is none). Skip the signal when
  reward/risk is below ``min_reward_risk``, the same rule the live bot uses.
* Enter at the next day's open, but only if reward/risk still holds at that
  price (a gap up can ruin it). Exit at the stop or the target; when both are
  inside one candle the stop is assumed to come first. A gap through either
  level fills at the open. With ``max_hold_days`` set, sell at that day's
  close if neither level was hit.

Costs and leverage follow Liquid's perps (liquid.trade, which routes to
Hyperliquid). Leverage is for this backtest only; the paper/live bot never
uses it.

* The position is ``leverage`` x your money, so gains and losses scale by it.
* Taker fees (0.095% per side) are charged on the full position on entry and
  exit, and funding (base rate 0.01% per 8 hours) on the full position for
  every day held.
* Maintenance margin is 1.25% of the position (Liquid's figure for 40x
  markets). If price falls to where only that is left (a ``1/leverage - 1.25%``
  drop) before the stop is reached, the position is liquidated at that price
  and you keep what remains. A gap straight through it loses everything.
* Once less than 1% of the starting money is left, the account stops trading.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from ..analysis.support_resistance import Bar, detect_levels, suggest_stop_take

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
    leverage: float = 1.0
    max_hold_days: Optional[int] = None     # None = hold until stop or target
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
    stop: float
    target: float
    support: float
    support_touches: int
    exit_time: datetime
    exit_price: float
    exit_reason: str        # stop | target | time | liquidated | end_of_data
    return_pct: float       # on your own money: after leverage, fees and financing
    days_held: int          # 1 = sold the same day it was bought


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


@dataclass
class _Position:
    entry_index: int
    entry_price: float
    signal: _Signal


def _find_signal(bars: List[Bar], t: int, p: BounceParams) -> Optional[_Signal]:
    bar = bars[t]
    if bar.close <= bar.open:
        return None
    sr = detect_levels(bars[t - p.lookback_bars + 1:t + 1],
                       window=p.sr_window, max_levels=p.max_levels)
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
    return _Signal(plan.stop_loss, plan.take_profit, support.price, support.touches)


def backtest_support_bounce(
    bars: List[Bar], symbol: str = "", params: Optional[BounceParams] = None
) -> BounceResult:
    """Run the support-bounce strategy over daily bars (oldest first)."""
    p = params or BounceParams()
    start = p.lookback_bars - 1
    if len(bars) < start + 2:
        return BounceResult(symbol)

    lev, fee = p.leverage, p.fee_pct
    # Price drop that leaves only the maintenance margin; 1x can't be liquidated.
    liq_move = 1 / lev - p.maintenance_margin if lev > 1 else None
    cash = 1.0
    trades: List[BounceTrade] = []
    equity: List[Tuple[datetime, float]] = []
    position: Optional[_Position] = None
    pending: Optional[_Signal] = None
    bars_in_market = 0

    def close_position(exit_index: int, price: float, reason: str) -> None:
        nonlocal cash, position
        assert position is not None
        days = exit_index - position.entry_index + 1
        move = price / position.entry_price - 1
        costs = lev * (fee * (2 + move) + p.funding_pct_per_day * days)
        ret = max(-1.0, lev * move - costs)
        cash *= 1 + ret
        s = position.signal
        trades.append(BounceTrade(
            symbol=symbol,
            entry_time=bars[position.entry_index].timestamp,
            entry_price=position.entry_price,
            stop=s.stop, target=s.target, support=s.support, support_touches=s.touches,
            exit_time=bars[exit_index].timestamp,
            exit_price=price,
            exit_reason=reason,
            return_pct=ret,
            days_held=days,
        ))
        position = None

    for t in range(start, len(bars)):
        bar = bars[t]

        if pending is not None:
            risk, reward = bar.open - pending.stop, pending.target - bar.open
            if risk > 0 and reward > 0 and reward >= p.min_reward_risk * risk:
                position = _Position(t, bar.open, pending)
            pending = None

        if position is not None:
            s = position.signal
            liq = position.entry_price * (1 - liq_move) if liq_move is not None else 0.0
            if bar.open <= liq:
                close_position(t, bar.open, "liquidated")
            elif bar.open <= s.stop:
                close_position(t, bar.open, "stop")
            elif bar.open >= s.target:
                close_position(t, bar.open, "target")
            elif bar.low <= max(s.stop, liq):
                # Falling price reaches whichever level is higher first.
                if s.stop >= liq:
                    close_position(t, s.stop, "stop")
                else:
                    close_position(t, liq, "liquidated")
            elif bar.high >= s.target:
                close_position(t, s.target, "target")
            elif p.max_hold_days and t - position.entry_index + 1 >= p.max_hold_days:
                close_position(t, bar.close, "time")

        if position is not None:
            bars_in_market += 1
            move = bar.close / position.entry_price - 1
            equity.append((bar.timestamp, cash * max(0.0, 1 + lev * move - fee * lev)))
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


def summarize(results: List[BounceResult]) -> Dict[str, Optional[float]]:
    """Pool all trades across symbols into headline numbers."""
    trades = [t for r in results for t in r.trades]
    returns = [t.return_pct for t in trades]
    gains = sum(r for r in returns if r > 0)
    losses = -sum(r for r in returns if r < 0)
    return {
        "symbols": len(results),
        "trades": len(trades),
        "win_rate": sum(1 for r in returns if r > 0) / len(returns) if returns else None,
        "avg_return": sum(returns) / len(returns) if returns else None,
        "profit_factor": gains / losses if losses else None,
        "beat_buy_hold": sum(1 for r in results if r.total_return > r.buy_hold_return),
        "liquidations": sum(r.liquidations for r in results),
        "wiped_out": sum(1 for r in results if r.equity and r.equity[-1][1] < WIPED_OUT),
        "avg_days_held": sum(t.days_held for t in trades) / len(trades) if trades else None,
    }
