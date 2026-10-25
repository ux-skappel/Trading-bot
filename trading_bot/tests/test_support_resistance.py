from datetime import datetime, timedelta

from trading_bot.analysis.event_classifier import EventLabel
from trading_bot.analysis.opportunity_scoring import ScoringInputs, score_opportunity
from trading_bot.analysis.support_resistance import (
    Bar,
    bars_from_csv_text,
    detect_levels,
    find_pivots,
    suggest_stop_take,
)
from trading_bot.connectors.market_data import MarketDataProvider, Quote


def _bars(closes, spread=0.0):
    start = datetime(2024, 1, 1)
    return [
        Bar(start + timedelta(days=i), c, c + spread, c - spread, c, 1_000_000)
        for i, c in enumerate(closes)
    ]


def _range_market(end_price: float):
    """Price bouncing between ~100 (floor) and ~110 (ceiling) three times."""
    leg_up = [100, 102, 104, 106, 108, 110]
    leg_down = [108, 106, 104, 102]
    closes = leg_up + leg_down + leg_up + leg_down + leg_up + leg_down
    closes += [100, 101, 102, 103, end_price]
    return _bars(closes, spread=0.2)


def test_find_pivots_on_zigzag():
    bars = _bars([1, 2, 3, 2, 1, 2, 3, 2, 1])
    pivots = find_pivots(bars, window=2)
    assert (2, 3) in pivots and (6, 3) in pivots   # swing highs
    assert (4, 1) in pivots                         # swing low


def test_flat_top_gives_one_pivot():
    bars = _bars([1, 2, 5, 5, 5, 2, 1])
    highs = [p for p in find_pivots(bars, window=2) if p[1] == 5]
    assert highs == [(2, 5)]


def test_range_market_finds_floor_and_ceiling():
    sr = detect_levels(_range_market(105), symbol="TEST", window=2)
    assert sr.nearest_support is not None and sr.nearest_resistance is not None
    assert abs(sr.nearest_support.price - 99.8) < 1.0
    assert abs(sr.nearest_resistance.price - 110.2) < 1.0
    assert sr.nearest_support.touches >= 2
    assert sr.nearest_resistance.touches >= 2


def test_price_above_all_levels_has_no_resistance():
    sr = detect_levels(_range_market(130), symbol="TEST", window=2)
    assert sr.resistances == []
    assert sr.supports
    assert "price discovery" in sr.summary()


def test_too_little_data_does_not_crash():
    assert detect_levels([], symbol="X").supports == []
    sr = detect_levels(_bars([10, 11]), symbol="X")
    assert sr.supports == [] and sr.resistances == []


def test_stop_below_support_and_target_below_resistance():
    sr = detect_levels(_range_market(103), symbol="TEST", window=2)
    t = suggest_stop_take(sr, entry=103, default_stop_pct=0.05,
                          default_take_pct=0.10, max_stop_pct=0.10)
    assert t.stop_loss < sr.nearest_support.price
    assert 103 < t.take_profit < sr.nearest_resistance.price
    assert t.reward_risk is not None and t.reward_risk > 1


def test_no_levels_falls_back_to_defaults():
    sr = detect_levels(_bars([10, 11]), symbol="X")
    t = suggest_stop_take(sr, entry=100, default_stop_pct=0.05,
                          default_take_pct=0.10, max_stop_pct=0.10)
    assert t.stop_loss == 95.0
    assert t.take_profit == 110.0
    assert t.reward_risk == 2.0


def _strong_buy_inputs(levels=None, price=150.0):
    return ScoringInputs(
        symbol="AAPL", asset_class="stock", sentiment=1.0, source_credibility=1.0,
        event_timestamp=datetime.utcnow(),
        event_label=EventLabel(event_type="earnings", bias=1, expected_impact=1.0),
        high_impact=None,
        quote=Quote(symbol="AAPL", price=price, prev_close=price / 1.03,
                    volume=4_000_000, avg_volume=1_500_000, daily_atr_pct=0.03, source="test"),
        duplicate_confirmations=3, is_in_universe=True, is_blacklisted=False,
        is_rumor=False, levels=levels,
    )


def test_buy_right_under_resistance_is_downgraded_to_watch():
    assert score_opportunity(_strong_buy_inputs()).direction == "buy"
    sr = detect_levels(_range_market(109.5), symbol="AAPL", window=2)
    out = score_opportunity(_strong_buy_inputs(levels=sr, price=109.5))
    assert out.direction == "watch"
    assert any("reward/risk" in r for r in out.reasoning)


def test_csv_semicolon_decimal_comma_newest_first():
    text = (
        "Dato;Åpning;Høy;Lav;Sluttkurs;Volum\n"
        "03.01.2024;101,5;103,0;100,0;102,0;1000\n"
        "02.01.2024;100,0;102,0;99,0;101,5;1200\n"
    )
    bars = bars_from_csv_text(text)
    assert [b.timestamp.day for b in bars] == [2, 3]
    assert bars[1].close == 102.0 and bars[1].high == 103.0


def test_csv_close_only():
    bars = bars_from_csv_text("date,price\n2024-01-01,10\n2024-01-02,11\n")
    assert bars[0].open == bars[0].high == bars[0].low == bars[0].close == 10


def test_chart_builds_and_drops_colliding_labels():
    from trading_bot.dashboard.sr_chart import _labelled, levels_frame, sr_chart

    bars = _range_market(105)
    sr = detect_levels(bars, symbol="TEST", window=2)
    spec = sr_chart(bars, sr, theme="dark").to_dict()
    assert len(spec["layer"]) == 8

    levels = levels_frame(sr)
    crowded = levels.copy()
    crowded.loc[len(crowded)] = ["Support", levels["price"].min() + 0.01, 0, 1, 0.01, bars[0].timestamp, "S x"]
    assert len(_labelled(crowded, y_range=12.0)) == len(levels)


def test_mock_history_is_deterministic_and_matches_quote():
    md = MarketDataProvider()
    a = md._mock_history("AAPL")
    b = md._mock_history("AAPL")
    assert [x.close for x in a] == [x.close for x in b]
    assert all(bar.low <= min(bar.open, bar.close) and bar.high >= max(bar.open, bar.close) for bar in a)
    quote = md.get_quote("NOT-A-REAL-TICKER-XYZ")
    assert quote is not None
    assert quote.price == md.get_history("NOT-A-REAL-TICKER-XYZ").bars[-1].close
