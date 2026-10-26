from datetime import datetime, timedelta

import pytest

from trading_bot.analysis.support_resistance import Bar
from trading_bot.copytrade.engine import CopyConfig, Post, replay

T0 = datetime(2026, 1, 1, 12, 0)
FREE = CopyConfig(fee_pct=0.0, funding_pct_per_day=0.0)       # exact math in tests


def minutes(path, symbol="SOL"):
    """Minute candles through the given prices; each candle opens at the previous close."""
    bars, prev = [], path[0]
    for i, price in enumerate(path):
        bars.append(Bar(T0 + timedelta(minutes=i), prev, max(prev, price), min(prev, price), price))
        prev = price
    return {symbol: bars}


def source(data):
    return lambda symbol, since: (data.get(symbol, []), "test")


def post(minute, text, mid=None, reply=None):
    return Post(mid or minute + 1, T0 + timedelta(minutes=minute, seconds=30), text, reply)


def test_long_hits_both_targets():
    data = minutes([100, 100, 102, 104, 106, 108, 111, 111])
    r = replay([post(0, "LONG $SOL entry 100 SL 95 TP1 105 TP2 110 10x")], source(data), FREE)
    t = r.trades[0]
    assert t.status == "closed"
    assert [f.reason for f in t.fills] == ["tp1", "tp2"]
    assert t.notional == pytest.approx(10_000)            # 10% margin x 10x
    assert t.pnl == pytest.approx(5_000 * 0.05 + 5_000 * 0.10)
    assert r.win_rate == 1.0


def test_stop_loss():
    data = minutes([100, 100, 98, 96, 94, 93])
    t = replay([post(0, "LONG SOL entry 100 SL 95 TP 110 10x")], source(data), FREE).trades[0]
    assert [f.reason for f in t.fills] == ["stop"]
    assert t.pnl == pytest.approx(-500)


def test_move_stop_to_entry_after_first_target():
    data = minutes([100, 100, 103, 106, 104, 102, 100, 99, 98])
    posts = [post(0, "LONG $SOL entry 100 SL 95 TP1 105 TP2 110 10x"), post(3, "Move SL to entry")]
    t = replay(posts, source(data), FREE).trades[0]
    assert [f.reason for f in t.fills] == ["tp1", "moved stop"]
    assert t.pnl == pytest.approx(250)


def test_his_close_half_then_target():
    data = minutes([100, 100, 102, 103, 104, 106, 111, 111])
    posts = [post(0, "LONG SOL @ 100 SL 95 TP 110 10x"), post(2, "Close half $SOL")]
    t = replay(posts, source(data), FREE).trades[0]
    assert [f.reason for f in t.fills] == ["his close", "tp1"]
    assert t.fills[0].fraction == pytest.approx(0.5)
    assert t.fills[0].price == 102                        # next candle's open after his post
    assert t.pnl == pytest.approx(5_000 * 0.02 + 5_000 * 0.10)


def test_high_leverage_short_is_liquidated_before_its_stop():
    data = minutes([100, 100, 101, 102, 103, 104, 106], symbol="BTC")
    t = replay([post(0, "SHORT BTC entry 100 SL 130 20x")], source(data), FREE).trades[0]
    assert t.fills[-1].reason == "liquidated"
    assert t.liquidation_price == pytest.approx(103.75)
    # Liquidated with only the 1.25% maintenance margin left: lose 75% of the $1,000 margin.
    assert t.pnl == pytest.approx(-750)


def test_waits_at_his_entry_when_price_ran_away():
    data = minutes([103, 103, 104, 102, 99.5, 101, 111])
    t = replay([post(0, "LONG SOL entry 100 SL 95 TP 110")], source(data), FREE).trades[0]
    assert t.entry_price == 100 and t.limit_price == 100
    assert t.status == "closed" and t.fills[-1].reason == "tp1"


def test_waiting_entry_expires():
    data = minutes([103] * 10)
    cfg = CopyConfig(fee_pct=0, funding_pct_per_day=0, entry_timeout_hours=5 / 60)
    t = replay([post(0, "LONG SOL entry 100 SL 95")], source(data), cfg).trades[0]
    assert t.status == "expired" and t.entry_price is None


def test_risk_percent_sizes_by_his_stop():
    data = minutes([100, 100, 97, 94])
    t = replay([post(0, "LONG SOL entry 100 SL 95 risk 1%")], source(data), FREE).trades[0]
    assert t.notional == pytest.approx(2_000)             # 1% of 10,000 / 5% stop
    assert t.pnl == pytest.approx(-100)


def test_close_without_coin_uses_the_replied_trade():
    data = {**minutes([100, 100, 101, 102, 103, 104]),
            **minutes([50, 50, 51, 52, 53, 54], symbol="ETH")}
    posts = [post(0, "LONG SOL @ 100 SL 90", mid=10), post(0, "LONG ETH @ 50 SL 45", mid=11),
             post(3, "closing here", mid=12, reply=10)]
    r = replay(posts, source(data), FREE)
    sol, eth = r.trades
    assert sol.status == "closed" and eth.status == "open"
    assert eth.unrealized > 0


def test_cancel_pending_entry():
    data = minutes([103] * 6)
    posts = [post(0, "LONG SOL entry 100 SL 95"), post(2, "Cancel SOL entry")]
    t = replay(posts, source(data), FREE).trades[0]
    assert t.status == "cancelled"


def test_chatter_creates_no_trades_and_is_explained():
    data = minutes([100] * 5)
    r = replay([post(0, "Good morning, market looks choppy")], source(data), FREE)
    assert r.trades == []
    assert r.posts[0].outcome.startswith("not a trade instruction")


def test_unknown_coin_is_skipped():
    r = replay([post(0, "LONG $NOPE entry 1 SL 0.9")], source({}), FREE)
    assert r.trades[0].status == "skipped"


def test_fees_and_funding_are_charged():
    data = minutes([100, 100, 104, 106, 111])
    t = replay([post(0, "LONG SOL @ 100 SL 95 TP 110 10x")], source(data), CopyConfig()).trades[0]
    gross = 10_000 * 0.10
    assert t.pnl < gross
    assert t.pnl == pytest.approx(gross - 10_000 * 0.00095 - 11_000 * 0.00095, rel=1e-3)
