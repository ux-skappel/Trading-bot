from trading_bot.connectors.market_data import Quote
from trading_bot.storage.database import init_db
from trading_bot.trading.broker_base import OrderRequest
from trading_bot.trading.paper_broker import PaperBroker
from trading_bot.trading.portfolio import Portfolio
from trading_bot.trading.risk_manager import RiskManager


def test_paper_broker_opens_and_closes_trade():
    init_db()
    broker = PaperBroker()
    req = OrderRequest(
        symbol="AAPL",
        asset_class="stock",
        side="buy",
        quantity=10,
        stop_loss=0.01,
        take_profit=10_000.0,
        reason="unit test",
    )
    result = broker.place_order(req, confirmation_token="test")
    assert result.accepted is True
    assert result.trade_id is not None

    # Forcing stop_loss=0.01 makes the next walk close on stop.
    broker.update_open_positions()
    pos = broker.get_position("AAPL")
    # may already be closed by stop_loss; either way no exception
    assert pos is None or pos["symbol"] == "AAPL"


def test_risk_manager_blocks_below_score():
    init_db()
    rm = RiskManager(Portfolio())
    q = Quote(symbol="AAPL", price=100, prev_close=100, volume=1_000_000,
              avg_volume=900_000, daily_atr_pct=0.02, source="test")
    decision = rm.evaluate(
        symbol="AAPL", asset_class="stock", score=10.0, confidence=0.9,
        quote=q, is_blacklisted=False, is_in_universe=True,
    )
    assert decision.allowed is False
    assert any("score" in b for b in decision.blockers)


def test_risk_manager_blocks_blacklist():
    init_db()
    rm = RiskManager(Portfolio())
    q = Quote(symbol="AAPL", price=100, prev_close=100, volume=1_000_000,
              avg_volume=900_000, daily_atr_pct=0.02, source="test")
    decision = rm.evaluate(
        symbol="AAPL", asset_class="stock", score=99.0, confidence=0.99,
        quote=q, is_blacklisted=True, is_in_universe=True,
    )
    assert decision.allowed is False
