import os
import tempfile
import textwrap

from trading_bot.backtesting.backtest import run_backtest


def _write(content: str) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    f.write(textwrap.dedent(content).lstrip())
    f.close()
    return f.name


def test_backtest_runs_on_minimal_inputs():
    events = _write(
        """
        timestamp,source,author,text,url
        2024-01-01 09:00:00,reuters,Reuters,Apple beats earnings $AAPL upgraded,https://x
        """
    )
    prices = _write(
        """
        timestamp,symbol,price
        2024-01-01 09:00:00,AAPL,150.0
        2024-01-01 10:00:00,AAPL,151.0
        2024-01-01 11:00:00,AAPL,152.0
        2024-01-01 12:00:00,AAPL,170.0
        2024-01-01 13:00:00,AAPL,165.0
        """
    )
    try:
        result = run_backtest(events, prices, horizon_bars=5)
        assert result.n_trades >= 0
    finally:
        os.unlink(events)
        os.unlink(prices)
