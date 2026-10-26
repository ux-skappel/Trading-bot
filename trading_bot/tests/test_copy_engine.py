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


def test_two_trades_in_one_post_and_selling_one_by_reply():
    data = {**minutes([100, 100, 101, 102, 103, 104], symbol="MU"),
            **minutes([50, 50, 51, 52, 53, 54], symbol="WDC")}
    posts = [post(0, "LONGER MICRON OG WESTERN NÅ 🔼🔼", mid=20), post(3, "SOLGT WESTERN 🤑", mid=21, reply=20)]
    r = replay(posts, source(data), FREE)
    mu, wdc = r.trades
    assert (mu.symbol, wdc.symbol) == ("MU", "WDC")
    assert wdc.status == "closed" and mu.status == "open"


def test_sold_all_by_reply_to_an_already_closed_trade_is_ignored():
    data = {**minutes([100, 100, 94, 93, 93, 93]), **minutes([50, 50, 51, 52, 53], symbol="ETH")}
    posts = [post(0, "SOL LONG 🔼 Bruker 10X giring og SL på $95", mid=30),
             post(0, "ETH LONG 🔼 Bruker 5X giring og ingen SL", mid=31),
             post(3, "SOLGT HELE NÅ", mid=32, reply=30)]
    r = replay(posts, source(data), FREE)
    sol, eth = r.trades
    assert sol.fills[-1].reason == "stop" and eth.status == "open"
    assert "he replied to #1 SOL (closed: stop)" in r.posts[-1].outcome


def test_market_max_leverage_caps_his_leverage_and_sets_margin():
    data = minutes([100, 100, 99, 97, 95.4, 95], symbol="AMD")
    r = replay([post(0, "AMD LONG 🔼 Bruker 20X giring og ingen SL")], source(data), FREE,
               max_leverage=lambda sym: 10.0)
    t = r.trades[0]
    assert t.leverage == 10 and "capped" in r.posts[0].outcome
    # 10x market: 5% maintenance margin -> liquidated after a 5% drop.
    assert t.liquidation_price == pytest.approx(95.0)
    assert t.fills[-1].reason == "liquidated"


def test_partial_position_scales_the_size():
    data = minutes([100, 100, 101], symbol="TSLA")
    t = replay([post(0, "TESLA LONG 🔼 Bruker 10X giring. Inne med 25% av full ønsket posisjon")],
               source(data), FREE).trades[0]
    assert t.notional == pytest.approx(10_000 * 0.25)


@pytest.mark.parametrize("words,margin", [
    ("Bruker 10X giring", 1_000),
    ("Risky! Bruker 10X giring", 500),
    ("Veldig risky. Bruker 10X giring", 250),
    ("Klink support! Bruker 10X giring", 1_500),
])
def test_his_tone_sets_the_size(words, margin):
    data = minutes([100, 100, 101], symbol="SOL")
    t = replay([post(0, f"SOL LONG 🔼 {words}")], source(data), FREE).trades[0]
    assert t.margin == pytest.approx(margin)


def test_tone_multipliers_come_from_the_config():
    data = minutes([100, 100, 101], symbol="SOL")
    cfg = CopyConfig(fee_pct=0, funding_pct_per_day=0, size_risky=0.3)
    t = replay([post(0, "SOL LONG 🔼 Risky. Inne med 50% av full posisjon. 10X")], source(data), cfg).trades[0]
    assert t.margin == pytest.approx(1_000 * 0.3 * 0.5)
    assert "x 30% (risky)" in t.note


def test_his_own_risk_percent_is_not_scaled_by_tone():
    data = minutes([100, 100, 101], symbol="SOL")
    t = replay([post(0, "LONG $SOL entry 100 SL 95 TP1 110 risk 1% klink support 10x")],
               source(data), FREE).trades[0]
    assert t.notional == pytest.approx(0.01 * 10_000 / 0.05)


def test_live_account_ignores_posts_from_before_it_started():
    data = minutes([100, 100, 101, 102, 103, 104, 105])
    posts = [post(0, "SOL LONG 🔼 Bruker 10X giring og SL på $95", mid=40),
             post(3, "ETH LONG 🔼 Bruker 5X giring", mid=41)]
    cfg = CopyConfig(fee_pct=0, funding_pct_per_day=0, live_since=T0 + timedelta(minutes=2))
    r = replay(posts, source({**data, **minutes([50] * 7, symbol="ETH")}), cfg)
    assert [t.symbol for t in r.trades] == ["ETH"]
    assert [o.post.message_id for o in r.posts] == [41]


def test_live_buy_seen_too_late_is_skipped_but_a_late_sell_still_happens():
    data = minutes([100] * 3 + [101, 102, 103, 104, 105] + [106] * 60)
    live = T0 - timedelta(hours=1)
    on_time = Post(50, T0 + timedelta(minutes=1), "SOL LONG 🔼 Bruker 10X giring og SL på $90",
                   received_at=T0 + timedelta(minutes=1, seconds=5))
    late_buy = Post(51, T0 + timedelta(minutes=2), "ETH LONG 🔼 Bruker 5X giring",
                    received_at=T0 + timedelta(minutes=50))
    late_sell = Post(52, T0 + timedelta(minutes=4), "SOLGT HELE NÅ", reply_to=50,
                     received_at=T0 + timedelta(minutes=45))
    cfg = CopyConfig(fee_pct=0, funding_pct_per_day=0, live_since=live)
    r = replay([on_time, late_buy, late_sell], source({**data, **minutes([50] * 70, symbol="ETH")}), cfg)
    assert [t.symbol for t in r.trades] == ["SOL"]
    assert any("missed" in o.outcome for o in r.posts)
    sol = r.trades[0]
    assert sol.status == "closed"
    assert sol.fills[-1].time == T0 + timedelta(minutes=45)   # sold when the bot saw his post
    assert sol.fills[-1].price == 106


def test_live_buy_fills_at_the_price_seen_when_the_post_arrived():
    data = minutes([100, 100, 101, 102, 103, 104])
    seen = Post(60, T0 + timedelta(minutes=1, seconds=10), "SOL LONG 🔼 Bruker 10X giring og SL på $95",
                received_at=T0 + timedelta(minutes=1, seconds=12))
    cfg = CopyConfig(fee_pct=0, funding_pct_per_day=0, live_since=T0)
    r = replay([seen], source(data), cfg, snapshot=lambda mid, sym: 100.37 if (mid, sym) == (60, "SOL") else None)
    t = r.trades[0]
    assert t.entry_price == 100.37
    assert t.entry_time == seen.received_at


def test_live_buy_later_than_two_minutes_is_skipped():
    data = minutes([100] * 10)
    late = Post(61, T0 + timedelta(minutes=1), "SOL LONG 🔼 Bruker 10X giring",
                received_at=T0 + timedelta(minutes=3, seconds=30))
    r = replay([late], source(data), CopyConfig(live_since=T0))
    assert r.trades == [] and "missed" in r.posts[0].outcome


def test_sell_replying_to_an_image_reply_finds_the_original_trade():
    data = minutes([100, 100, 101, 102, 103, 104, 105], symbol="SPCX")
    posts = [post(0, "SPACEX LONG 🔼 Bruker 10X giring og SL $90", mid=70),
             post(2, "", mid=71, reply=70),                              # chart image replying to the trade
             post(4, "BOOM 🔼🔼 SELGER HELE NÅ", mid=72, reply=71)]       # sell replying to the image
    r = replay(posts, source(data), FREE)
    t = r.trades[0]
    assert t.status == "closed" and t.fills[-1].reason == "his close"
    assert "image" in r.posts[1].outcome


def test_ignored_sell_says_which_trade_he_replied_to():
    data = minutes([100, 100, 94, 93, 93, 93], symbol="SPCX")
    posts = [post(0, "SPACEX LONG 🔼 Bruker 10X giring og SL $95", mid=80),
             post(3, "SELGER HELE NÅ", mid=81, reply=80)]
    r = replay(posts, source(data), FREE)
    assert "he replied to #1 SPCX (closed: stop)" in r.posts[-1].outcome


def test_trade_without_free_margin_is_reported_as_not_bought():
    data = {**minutes([100] * 6), **minutes([50] * 6, symbol="ETH")}
    posts = [post(0, "SOL LONG 🔼 Bruker 10X giring, 100% av porteføljen", mid=90),
             post(1, "ETH LONG 🔼 Bruker 10X giring", mid=91)]
    r = replay(posts, source(data), FREE)
    assert r.trades[1].status == "skipped"
    assert "not bought" in r.posts[1].outcome


def _two_open(image_text):
    data = {**minutes([300, 300, 301, 302, 303, 304], symbol="GOOGL"),
            **minutes([70, 70, 69, 68, 67, 66], symbol="HYPE")}
    posts = [post(0, "GOOGLE LONG 🔼 Bruker 20X giring", mid=100),
             post(0, "HYPE SHORT 🔽 Bruker 10X giring", mid=101),
             Post(102, T0 + timedelta(minutes=3, seconds=30), "BOOM 🔼🔼 SELGER HELE NÅ", image_text=image_text)]
    return replay(posts, source(data), FREE)


def test_sell_with_image_closes_the_coin_named_in_the_image():
    r = _two_open("Liquid\nGOOGL-USD\nLong 20x\nEntry 300.00\nROE +40.1%")
    googl, hype = r.trades
    assert googl.status == "closed" and hype.status == "open"
    assert "coin read from his image" in r.posts[-1].outcome


def test_sell_with_image_naming_several_positions_does_not_guess():
    r = _two_open("Portfolio\nGOOGL +40%\nHYPE +12%")
    assert all(t.status == "open" for t in r.trades)
    assert "not guessing" in r.posts[-1].outcome


def test_sell_whose_image_names_a_coin_we_dont_hold_closes_nothing():
    r = _two_open("Liquid\nAMD-USD\nLong 10x\nROE +31%")
    assert all(t.status == "open" for t in r.trades)
    assert "image says AMD, which the bot doesn't hold" in r.posts[-1].outcome


def test_sell_naming_a_coin_we_dont_hold_does_not_close_the_latest():
    data = minutes([90, 90, 91, 92, 93], symbol="HOOD")
    posts = [post(0, "HOOD LONG 🔼 Bruker 10X giring", mid=140),
             Post(141, T0 + timedelta(minutes=3), "Fin long på AMD 🔼 Selger hele nå",
                  image_text="Liquid | AMD PERP 10x LONG")]
    r = replay(posts, source(data), FREE)
    assert r.trades[0].status == "open"
    assert "no AMD position was ever copied" in r.posts[-1].outcome


def test_sell_without_image_text_falls_back_to_latest_position():
    r = _two_open(None)
    googl, hype = r.trades
    assert hype.status == "closed" and googl.status == "open"


def test_reply_sell_shows_what_his_image_says():
    data = {**minutes([200, 200, 205, 210, 215], symbol="NVDA"), **minutes([370] * 5, symbol="TSLA")}
    posts = [post(0, "NVIDIA LONG 🔼 Bruker 10X giring", mid=110),
             post(0, "TESLA LONG 🔼 Bruker 10X giring", mid=111),
             Post(112, T0 + timedelta(minutes=3, seconds=30), "SELGER HELE NÅ", reply_to=110,
                  image_text="Liquid | NVDA PERP 10x LONG | CURRENT RETURN | +22.61%"),
             Post(113, T0 + timedelta(minutes=3, seconds=40), "SELGER HELE NÅ", reply_to=111,
                  image_text="Liquid | NVDA PERP 10x LONG")]
    r = replay(posts, source(data), FREE)
    nvda_sell, tsla_sell = r.posts[-2].outcome, r.posts[-1].outcome
    assert "#1 NVDA" in nvda_sell and "he replied to it" in nvda_sell and "image says NVDA" in nvda_sell
    assert "#2 TSLA" in tsla_sell and "⚠ his image says NVDA" in tsla_sell


def test_history_test_sizes_each_trade_on_its_own():
    data = {**minutes([100] * 6), **minutes([50] * 6, symbol="ETH")}
    posts = [post(0, "SOL LONG 🔼 Bruker 10X giring, 100% av porteføljen", mid=120),
             post(1, "ETH LONG 🔼 Bruker 10X giring", mid=121)]
    shared = replay(posts, source(data), FREE)
    alone = replay(posts, source(data), CopyConfig(fee_pct=0, funding_pct_per_day=0, share_margin=False))
    assert shared.trades[1].status == "skipped"
    assert alone.trades[1].status == "open" and alone.trades[1].notional == pytest.approx(10_000)


def test_discussion_topic_post_never_buys():
    data = minutes([300] * 5, symbol="TSLA")
    p = Post(130, T0 + timedelta(minutes=1), "TSLA LONG 🔼 Bruker 10X giring", sell_only=True)
    r = replay([p], source(data), FREE)
    assert r.trades == [] and "not bought" in r.posts[0].outcome


def test_discussion_topic_sell_with_picture_closes_that_position():
    data = {**minutes([300, 300, 305, 310, 315], symbol="TSLA"), **minutes([200] * 5, symbol="NVDA")}
    posts = [post(0, "TESLA LONG 🔼 Bruker 10X giring", mid=131), post(0, "NVIDIA LONG 🔼 Bruker 10X giring", mid=132),
             Post(133, T0 + timedelta(minutes=3, seconds=30), "SELGER HELE NÅ", sell_only=True,
                  image_text="Liquid | TSLA | PERP 10x LONG | CURRENT RETURN | +50%")]
    r = replay(posts, source(data), FREE)
    tsla, nvda = r.trades
    assert tsla.status == "closed" and nvda.status == "open"


def test_discussion_topic_sell_without_coin_is_not_guessed():
    data = {**minutes([300] * 5, symbol="TSLA"), **minutes([200] * 5, symbol="NVDA")}
    posts = [post(0, "TESLA LONG 🔼 Bruker 10X giring", mid=134), post(0, "NVIDIA LONG 🔼 Bruker 10X giring", mid=135),
             Post(136, T0 + timedelta(minutes=3, seconds=30), "SELGER HELE NÅ", sell_only=True)]
    r = replay(posts, source(data), FREE)
    assert all(t.status == "open" for t in r.trades)
    assert "not guessing" in r.posts[-1].outcome
