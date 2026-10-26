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
    r = _two_open("• Liquid | GOOGL PERP 20x LONG | CURRENT RETURN | +40.1% | ENTRY | $300,00 | MARK | $302,00")
    googl, hype = r.trades
    assert googl.status == "closed" and hype.status == "open"
    assert "his trade card shows it" in r.posts[-1].outcome


def test_sell_with_image_naming_several_positions_does_not_guess():
    r = _two_open("Portfolio\nGOOGL +40%\nHYPE +12%")
    assert all(t.status == "open" for t in r.trades)
    assert "not guessing" in r.posts[-1].outcome


def test_sell_whose_image_names_a_coin_we_dont_hold_closes_nothing():
    r = _two_open("• Liquid | AMD PERP 10x LONG | CURRENT RETURN | +31% | ENTRY | $500,00 | MARK | $510,00")
    assert all(t.status == "open" for t in r.trades)
    assert "card shows AMD 10x long, not a position the bot copied" in r.posts[-1].outcome


def test_sell_naming_a_coin_we_dont_hold_does_not_close_the_latest():
    data = minutes([90, 90, 91, 92, 93], symbol="HOOD")
    posts = [post(0, "HOOD LONG 🔼 Bruker 10X giring", mid=140),
             Post(141, T0 + timedelta(minutes=3), "Fin long på AMD 🔼 Selger hele nå",
                  image_text="Liquid | AMD PERP 10x LONG")]
    r = replay(posts, source(data), FREE)
    assert r.trades[0].status == "open"
    assert "no AMD position was ever copied" in r.posts[-1].outcome


def test_review_file_lists_posts_with_what_the_bot_did(tmp_path):
    from trading_bot.main import _write_copy_review
    data = minutes([100, 100, 102, 104])
    posts = [post(0, "SOL LONG 🔼 Bruker 10X giring", mid=150), post(1, "", mid=151),
             Post(152, T0 + timedelta(minutes=2), "Noen som har sett TSLA i dag?", sell_only=True)]
    path = tmp_path / "review.txt"
    _write_copy_review(replay(posts, source(data), FREE), path)
    text = path.read_text(encoding="utf-8")
    assert "SOL LONG 🔼 Bruker 10X giring" in text and ">>> bot did: #1 opened" in text
    assert "Noen som har sett TSLA" not in text            # discussion chat is left out
    assert "#1 SOL long 10x open" in text


def test_sell_without_image_text_falls_back_to_latest_position():
    r = _two_open(None)
    googl, hype = r.trades
    assert hype.status == "closed" and googl.status == "open"


def test_reply_sell_shows_what_his_image_says():
    data = {**minutes([200, 200, 205, 210, 215], symbol="NVDA"), **minutes([370] * 5, symbol="TSLA")}
    posts = [post(0, "NVIDIA LONG 🔼 Bruker 10X giring", mid=110),
             post(0, "TESLA LONG 🔼 Bruker 10X giring", mid=111),
             Post(112, T0 + timedelta(minutes=3, seconds=30), "SELGER HELE NÅ", reply_to=110,
                  image_text="• Liquid | NVDA PERP 10x LONG | CURRENT RETURN | +22.61%"),
             Post(113, T0 + timedelta(minutes=3, seconds=40), "SELGER HELE NÅ", reply_to=111,
                  image_text="• Liquid | NVDA PERP 10x LONG")]
    r = replay(posts, source(data), FREE)
    nvda_sell, tsla_sell = r.posts[-2].outcome, r.posts[-1].outcome
    assert "#1 NVDA" in nvda_sell and "he replied to it" in nvda_sell and "card: NVDA 10x long" in nvda_sell
    assert "#2 TSLA" in tsla_sell and "⚠ his card shows NVDA 10x long" in tsla_sell


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


# --- Lessons from reading all his posts (paraphrased examples) ---------------------------

def hours(rows, symbol="NVDA", start=T0):
    """Hourly candles from (open, high, low, close) rows."""
    return {symbol: [Bar(start + timedelta(hours=i), *row) for i, row in enumerate(rows)]}


def test_a_stop_move_inside_an_hour_comes_before_that_hours_low():
    # He buys at 12:00, moves his stop to 214 at 13:17; the 13:00 candle later dips to 209
    # (below the 20x liquidation). The stop move came first, so the copy is stopped at 214.
    data = hours([(214, 215, 213, 214.5), (214.5, 215.5, 214.2, 215.3), (215.3, 215.9, 209, 209.6),
                  (209.6, 210, 208.6, 209.7)], start=T0 - timedelta(hours=1))
    posts = [Post(1, T0 - timedelta(minutes=42), "NVIDIA LONG 🔼 Bruker 20X giring og ingen SL"),
             Post(2, T0 + timedelta(hours=1, minutes=17), "NVIDIA SL til $214", reply_to=1)]
    t = replay(posts, source(data), FREE, now=T0 + timedelta(hours=4)).trades[0]
    assert [f.reason for f in t.fills] == ["moved stop"] and t.fills[0].price == 214


def test_no_buy_when_prices_only_start_weeks_later():
    data = minutes([45, 45, 46], symbol="IREN")
    late = Post(1, T0 - timedelta(days=20), "IREN SHORT 🔽 Bruker 10X giring og SL på $39,1")
    t = replay([late], source(data), FREE).trades[0]
    assert t.status == "skipped" and "no price from when he posted" in t.note


def test_no_buy_when_the_price_is_already_past_his_stop():
    data = minutes([100, 100, 99])
    t = replay([post(0, "SOL SHORT 🔽 Bruker 20X giring og SL på $99,5")], source(data), FREE).trades[0]
    assert t.status == "skipped" and "already past his stop" in t.note


def test_his_levels_follow_the_markets_units():
    # kPEPE-style: he writes 0,00521 for a coin the feed prices at 0.00000500.
    data = minutes([5.0e-6, 5.0e-6, 5.1e-6, 5.2e-6], symbol="PEPE")
    t = replay([post(0, "PEPE SHORT 🔽 Bruker 10X giring og SL på $0,00521")], source(data), FREE).trades[0]
    assert t.stop == pytest.approx(5.21e-6)


def test_added_position_inherits_coin_and_leverage_from_the_reply():
    data = minutes([300, 300, 301, 302, 303], symbol="TSLA")
    posts = [post(0, "TESLA LONG 🔼 Bruker 10X giring", mid=60),
             post(2, "Longet mer ⚡️⚡️", mid=61, reply=60)]
    first, added = replay(posts, source(data), FREE).trades
    assert (added.symbol, added.side, added.leverage) == ("TSLA", "long", 10)
    assert "adds to his TSLA position" in " ".join(added.signal.notes)


def test_leverage_comes_from_his_card_when_the_post_has_none():
    data = minutes([590, 590, 591], symbol="META")
    p = Post(70, T0 + timedelta(seconds=30), "Velger long META",
             image_text="• Liquid | METADAO PERP 20x LONG | CURRENT RETURN | -0.1% | ENTRY | $590,00 | MARK | $589,50")
    t = replay([p], source(data), FREE).trades[0]
    assert t.leverage == 20


def test_discussion_sell_of_a_different_position_closes_nothing():
    # His 10x short is copied; in the discussion topic he sells a separate 75x LONG in the same coin.
    data = minutes([9.5, 9.5, 9.4, 9.3, 9.2], symbol="LINK")
    posts = [post(0, "LINK SHORT 🔽 Bruker 10X giring og SL på $9,81", mid=80),
             Post(81, T0 + timedelta(minutes=3), "Solgt", sell_only=True,
                  image_text="• Liquid | Chainlink PERP 75x LONG | CURRENT RETURN | +88% | ENTRY | $9,40 | MARK | $9,52")]
    r = replay(posts, source(data), FREE)
    assert r.trades[0].status == "open" and "75x long, not a position the bot copied" in r.posts[-1].outcome


def test_selling_his_100x_keeps_the_40x_copy():
    data = minutes([77400, 77400, 77300, 77200, 77100], symbol="BTC")
    posts = [post(0, "BTC SHORT 🔽 Bruker 40X giring og SL på $78300", mid=90),
             Post(91, T0 + timedelta(minutes=3), "Solgt min 100x nå\n\nBeholder 40x", sell_only=True,
                  image_text="• Liquid | Bitcoin PERP | 100x SHORT | CURRENT RETURN | +55% | ENTRY | $77 390 | MARK | $77 010")]
    r = replay(posts, source(data), FREE)
    assert r.trades[0].status == "open"


def test_old_posts_close_at_the_mark_on_his_card():
    data = minutes([300, 300, 301, 302, 303], symbol="TSLA")
    posts = [post(0, "TESLA LONG 🔼 Bruker 10X giring", mid=100),
             Post(101, T0 + timedelta(minutes=2, seconds=30), "Selger hele nå", reply_to=100,
                  image_text="• Liquid | TSLA | PERP 10x LONG | CURRENT RETURN | +5% | ENTRY | $300,00 | MARK | $301,50")]
    t = replay(posts, source(data), FREE).trades[0]
    assert t.fills[-1].price == 301.5


def test_sizes_grow_with_the_account_after_a_win():
    data = {**minutes([100, 100, 100, 110, 110, 110, 110]), **minutes([50] * 7, symbol="ETH")}
    posts = [post(0, "SOL LONG 🔼 Bruker 10X giring", mid=200),
             post(3, "Selger hele nå", mid=201, reply=200),       # sold at 110
             post(4, "ETH LONG 🔼 Bruker 10X giring", mid=202)]
    for share in (True, False):   # live and the test on old posts size the same way
        sol, eth = replay(posts, source(data), CopyConfig(fee_pct=0, funding_pct_per_day=0,
                                                         share_margin=share)).trades
        assert sol.pnl == pytest.approx(1_000)                 # +10% on a $10,000 position
        assert eth.margin == pytest.approx(0.10 * 11_000)      # 10% of the new $11,000


def _month_end_run(withdraw, sol_path):
    start = datetime(2026, 1, 31, 23, 50)
    def bars(path):
        out, prev = [], path[0]
        for i, price in enumerate(path):
            out.append(Bar(start + timedelta(minutes=i), prev, max(prev, price), min(prev, price), price))
            prev = price
        return out
    data = {"SOL": bars(sol_path), "ETH": bars([50] * len(sol_path))}
    posts = [Post(1, start + timedelta(seconds=30), "SOL LONG 🔼 Bruker 10X giring"),
             Post(2, start + timedelta(minutes=5, seconds=30), "Selger hele nå", reply_to=1),
             Post(3, start + timedelta(minutes=15, seconds=30), "ETH LONG 🔼 Bruker 10X giring")]   # Feb 1, 00:05
    cfg = CopyConfig(fee_pct=0, funding_pct_per_day=0, withdraw_monthly=withdraw)
    return replay(posts, source(data), cfg, now=start + timedelta(minutes=20))


def test_profit_is_taken_out_at_the_month_end():
    r = _month_end_run(True, [100] * 3 + [110] * 17)            # SOL +10% at 10x: +$1,000 in January
    sol, eth = r.trades
    assert sol.pnl == pytest.approx(1_000)
    assert r.withdrawals == [(datetime(2026, 2, 1), pytest.approx(1_000))]
    assert eth.margin == pytest.approx(1_000)                    # February starts from $10,000 again
    assert r.account_now == pytest.approx(10_000) and r.realized_pnl == pytest.approx(1_000)


def test_a_losing_month_takes_nothing_out():
    r = _month_end_run(True, [100] * 3 + [95] * 17)
    assert r.withdrawals == [] and r.trades[1].margin == pytest.approx(0.10 * 9_500)


def test_withdrawals_can_be_switched_off():
    r = _month_end_run(False, [100] * 3 + [110] * 17)
    assert r.withdrawals == [] and r.trades[1].margin == pytest.approx(1_100)


def test_going_short_closes_his_long_in_the_same_coin():
    # He never posts a sell for the long, but a short in the same coin can't sit next to it on Liquid.
    data = minutes([100, 100, 104, 108, 108, 108])
    posts = [post(0, "SOL LONG 🔼 Bruker 10X giring", mid=300),
             post(3, "SOL SHORT 🔽 Bruker 10X giring og SL på $112", mid=301)]
    long, short = replay(posts, source(data), FREE).trades
    assert long.status == "closed" and long.fills[-1].reason == "flipped"
    assert long.fills[-1].price == short.entry_price and short.status == "open"


def test_a_stop_moved_inside_an_hour_is_not_hit_by_that_hours_open():
    # Short at the 15:00 open (80,180); at 15:34 he moves the stop to 79,900 in profit. The 15:00
    # open came before the move, so it can't stop the trade at the entry price.
    start = datetime(2026, 10, 24, 14, 0)
    data = hours([(80300, 80400, 80100, 80180), (80180, 80250, 79400, 79500), (79500, 79600, 79000, 79100)],
                 symbol="BTC", start=start)
    posts = [Post(1, start + timedelta(minutes=47), "BTC SHORT 🔽 Bruker 40X giring og SL på $80.6K"),
             Post(2, start + timedelta(hours=1, minutes=34), "BOOM 🔽\n\nSL i profitt på $79900", reply_to=1)]
    t = replay(posts, source(data), FREE, now=start + timedelta(hours=4)).trades[0]
    assert t.entry_price == 80180
    assert t.fills[-1].reason == "moved stop" and t.fills[-1].price == 79900 and t.pnl > 0


def test_rounded_card_prices_are_not_used():
    from trading_bot.copytrade.images import cards
    card = cards("• Liquid | DOGECOIN | PERP 10x LONG | CURRENT RETURN | +30% | ENTRY | $0,09 | MARK | $0,09")[0]
    assert (card.symbol, card.entry, card.mark) == ("DOGE", None, None)
    card = cards("• Liquid | SOLANA | PERP 10x LONG | CURRENT RETURN | +5% | ENTRY | $98,23 | MARK | $100,76")[0]
    assert (card.entry, card.mark) == (98.23, 100.76)


def test_old_calls_use_the_entry_on_his_reply_card():
    data = minutes([100, 100, 101, 102, 103], symbol="SOL")
    posts = [post(0, "SOL LONG 🔼 Bruker 10X giring", mid=400),
             Post(401, T0 + timedelta(minutes=2), "🔼🔼🔼", reply_to=400,
                  image_text="• Liquid | SOLANA | PERP 10x LONG | CURRENT RETURN | +5% | ENTRY | $99,50 | MARK | $101,20")]
    assert replay(posts, source(data), FREE).trades[0].entry_price == 99.5
    live = CopyConfig(fee_pct=0, funding_pct_per_day=0, live_since=T0)
    assert replay(posts, source(data), live).trades[0].entry_price == 100   # live: the price when seen
