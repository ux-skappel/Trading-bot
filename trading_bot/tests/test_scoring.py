from datetime import datetime

from trading_bot.analysis.event_classifier import EventLabel
from trading_bot.analysis.opportunity_scoring import ScoringInputs, score_opportunity
from trading_bot.connectors.market_data import Quote


def _quote(symbol="AAPL", price=150.0, change=0.01, atr=0.02):
    return Quote(
        symbol=symbol,
        price=price,
        prev_close=price / (1 + change),
        volume=2_000_000,
        avg_volume=1_500_000,
        daily_atr_pct=atr,
        source="test",
    )


def test_strong_positive_news_scores_high():
    inp = ScoringInputs(
        symbol="AAPL",
        asset_class="stock",
        sentiment=0.9,
        source_credibility=0.95,
        event_timestamp=datetime.utcnow(),
        event_label=EventLabel(event_type="earnings", bias=1, expected_impact=0.8),
        high_impact=None,
        quote=_quote(),
        duplicate_confirmations=2,
        is_in_universe=True,
        is_blacklisted=False,
        is_rumor=False,
    )
    out = score_opportunity(inp)
    assert out.total_score >= 65.0
    assert out.direction in {"buy", "watch"}


def test_blacklist_forces_avoid():
    inp = ScoringInputs(
        symbol="AAPL", asset_class="stock", sentiment=0.9, source_credibility=0.95,
        event_timestamp=datetime.utcnow(),
        event_label=EventLabel(event_type="earnings", bias=1, expected_impact=0.8),
        high_impact=None, quote=_quote(),
        duplicate_confirmations=3, is_in_universe=True,
        is_blacklisted=True, is_rumor=False,
    )
    assert score_opportunity(inp).direction == "avoid"


def test_rumor_lowers_confidence():
    inp = ScoringInputs(
        symbol="AAPL", asset_class="stock", sentiment=0.3, source_credibility=0.4,
        event_timestamp=datetime.utcnow(),
        event_label=None, high_impact=None, quote=_quote(),
        duplicate_confirmations=0, is_in_universe=True,
        is_blacklisted=False, is_rumor=True,
    )
    out = score_opportunity(inp)
    assert out.confidence < 0.5
    assert out.direction in {"watch", "avoid"}
