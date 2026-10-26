import math
from dataclasses import replace
from datetime import datetime, timedelta

from trading_bot.analysis.support_resistance import Bar
from trading_bot.backtesting.support_bounce import (
    BounceParams,
    backtest_support_bounce,
    summarize,
)

P = BounceParams(lookback_bars=30, sr_window=2, min_touches=2, min_reward_risk=1.5, fee_pct=0.0)
CYCLE = [100, 102, 104, 106, 108, 110, 108, 106, 104, 102]   # floor ~100, ceiling ~110


def _candles(closes, wick=0.2):
    """Gap-free candles: each opens at the previous close."""
    start = datetime(2024, 1, 1)
    bars, prev = [], closes[0]
    for i, c in enumerate(closes):
        bars.append(Bar(start + timedelta(days=i), prev, max(prev, c) + wick, min(prev, c) - wick, c, 1e6))
        prev = c
    return bars


def test_bounces_in_a_range_hit_the_target():
    r = backtest_support_bounce(_candles(CYCLE * 5), "RANGE", P)
    assert len(r.trades) == 2
    assert all(t.exit_reason == "target" for t in r.trades)
    assert all(t.support_touches >= 2 for t in r.trades)
    assert r.win_rate == 1.0
    assert math.isclose(r.total_return + 1, math.prod(1 + t.return_pct for t in r.trades))


def test_broken_support_exits_at_the_stop():
    closes = CYCLE * 3 + [100, 102, 101, 97, 94, 92, 90]
    r = backtest_support_bounce(_candles(closes), "BREAK", P)
    assert len(r.trades) == 1
    t = r.trades[0]
    assert t.exit_reason == "stop"
    assert t.stop < t.support
    assert math.isclose(t.return_pct, t.stop / t.entry_price - 1)
    assert -0.05 < t.return_pct < 0


def test_gap_below_stop_fills_at_the_open():
    bars = _candles(CYCLE * 3 + [100, 102, 101, 97, 94])
    gap = bars[33]
    bars[33] = replace(gap, open=95.0, high=96.0, low=94.0, close=95.5)
    t = backtest_support_bounce(bars, "GAP", P).trades[0]
    assert t.exit_reason == "stop"
    assert t.exit_price == 95.0


def test_no_look_ahead():
    bars = _candles(CYCLE * 5)
    full = backtest_support_bounce(bars, "X", P).trades
    for k in range(P.lookback_bars + 1, len(bars)):
        prefix = backtest_support_bounce(bars[:k], "X", P).trades
        closed = [(t.entry_time, t.exit_time) for t in prefix if t.exit_reason != "end_of_data"]
        expected = [(t.entry_time, t.exit_time) for t in full if t.exit_time <= bars[k - 1].timestamp]
        assert closed == expected, f"differs when cut at bar {k}"


def test_too_little_data_gives_empty_result():
    r = backtest_support_bounce(_candles(CYCLE), "SHORT", P)
    assert r.trades == [] and r.total_return == 0.0 and r.win_rate is None


def test_buy_hold_and_summary():
    bars = _candles(CYCLE * 5)
    r = backtest_support_bounce(bars, "RANGE", P)
    start = bars[P.lookback_bars - 1].close
    assert math.isclose(r.buy_hold_return, bars[-1].close / start - 1)
    s = summarize([r, backtest_support_bounce(_candles(CYCLE), "SHORT", P)])
    assert s["symbols"] == 2 and s["trades"] == 2
    assert s["win_rate"] == 1.0 and s["profit_factor"] is None
