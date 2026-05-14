# Trading Bot — MVP (paper trading)

A safe MVP for a trading **intelligence** system. It monitors public news /
RSS / Reddit / X feeds, classifies whether events may move specific stocks or
crypto assets, ranks opportunities with a transparent 0–100 score, and
**simulates** trades in a paper portfolio.

> ⚠️ **Not financial advice.** This software ships with **live trading
> structurally disabled.** The only broker wired in is the internal paper
> broker. Even if you flip `LIVE_TRADING=true`, no code path exists to send a
> real order — you would still have to write and wire in a live broker
> integration yourself.

## Features

- **Ingestion** — RSS, X (official API), Reddit (official API).
- **Normalization** — timestamp / source / author / text / tickers / crypto /
  companies / confidence.
- **High-impact accounts watchlist** — Trump, White House, US Treasury, Fed,
  ECB, SEC, CFTC, FTC, EU Commission, Musk, Tim Cook, Saylor, Vitalik, CZ,
  Brian Armstrong, official company accounts. Topic→sector mapping for
  tariffs, sanctions, crypto regulation, rates, antitrust, defense
  spending, energy, taxes, geopolitics. Tone classifier distinguishes
  official statements, threats, jokes, campaign rhetoric, rumors.
- **Event classifier** — earnings, lawsuit, FDA, contracts, M&A, bankruptcy,
  hacks, ETF, regulation, partnership, CEO change, production, geopolitics,
  analyst calls, monetary policy.
- **Credibility filter** — per-source weight, double-confirmation logic,
  rumor flagging, official-account override.
- **Transparent scorer** — 10 weighted factors, every contribution is
  recorded and visible in the dashboard.
- **Paper broker** — entry / stop loss / take profit / trailing stop /
  exit reason / PnL.
- **Risk manager** — position size cap, daily loss cap, max open positions,
  asset-class exposure cap, cooldown after loss, blacklist, extreme-vol gate.
- **Human-approval mode** — bot creates a pending trade and a recommendation;
  you approve or reject in the dashboard.
- **Streamlit dashboard** — news / opportunities / portfolio / open positions
  (with approve buttons) / performance / decision logs / sources / watchlists
  / config.
- **Backtester** — replay events + price CSVs, get win rate, average return,
  max drawdown, Sharpe-ish, best/worst trades.

## Project layout

```
trading_bot/
  app.py                  # pipeline orchestrator
  main.py                 # CLI entry point
  config.py               # all settings + safety flags
  connectors/
    base.py
    rss_feeds.py
    x_api.py
    reddit_api.py
    market_data.py
  analysis/
    entity_extraction.py
    sentiment.py
    event_classifier.py
    high_impact.py        # politicians/regulators/CEOs etc.
    opportunity_scoring.py
  trading/
    broker_base.py
    paper_broker.py
    portfolio.py
    risk_manager.py
    strategy.py
  storage/
    database.py
    models.py
  dashboard/
    streamlit_app.py
  backtesting/
    backtest.py
  data/
    sample_events.csv
    sample_prices.csv
  tests/
.env.example
requirements.txt
README.md
```

## Run locally

1. **Clone & venv:**
   ```bash
   git clone <repo>
   cd Trading-bot
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   pip install -e .
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   # edit .env — leave LIVE_TRADING=false
   ```

3. **One-shot scan:**
   ```bash
   python -m trading_bot.main run
   ```

4. **Loop mode (every 5 minutes):**
   ```bash
   python -m trading_bot.main loop --every 300
   ```

5. **Dashboard:**
   ```bash
   streamlit run trading_bot/dashboard/streamlit_app.py
   ```
   Then open <http://localhost:8501>.

6. **Approve a pending trade from the CLI:**
   ```bash
   python -m trading_bot.main approve 7 --approver=ux
   ```

7. **Run the tests:**
   ```bash
   pytest -q
   ```

8. **Backtest with the bundled sample data:**
   ```bash
   python -m trading_bot.main backtest \
     --events trading_bot/data/sample_events.csv \
     --prices trading_bot/data/sample_prices.csv \
     --horizon 5
   ```

## Adding real API keys

All keys live in `.env`. **Never** commit `.env`.

| Source | Env vars | Notes |
|---|---|---|
| X / Twitter | `X_BEARER_TOKEN` (+ optional `X_API_KEY/SECRET`, `X_ACCESS_TOKEN/SECRET`) | Official API v2 only. Without a token, the X connector returns no items. |
| Reddit | `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT`, `REDDIT_SUBREDDITS` | Create a "script" app at <https://www.reddit.com/prefs/apps>. |
| RSS | n/a | Edit `Settings.rss_feeds` in `trading_bot/config.py` or extend `RSSConnector` to read from env. |
| Market data | default yfinance (no key). For a paid provider, set `MARKET_DATA_PROVIDER` and `MARKET_DATA_API_KEY` and add a new `MarketDataProvider` branch. |
| Alpaca (future) | `ALPACA_API_KEY`, `ALPACA_API_SECRET`, `ALPACA_BASE_URL` | Live broker NOT shipped; you'd need to add an `AlpacaBroker(BrokerBase)`. |

After updating `.env`, restart the process.

## Safe paper-trading workflow

The defaults are deliberately conservative:

- `LIVE_TRADING=false`
- `TRADING_MODE=paper`
- `HUMAN_APPROVAL_MODE=true`
- `REQUIRE_MANUAL_CONFIRMATION=true`
- broker = internal `PaperBroker`

In this state the bot will:

1. Ingest news / X / Reddit (RSS works offline if no keys are set).
2. Normalize, classify, score every event.
3. For any candidate that clears the score + risk gates, **create a
   `pending_approval` trade** and a `Recommendation` row.
4. **Never auto-fill** the trade. You must approve from the Streamlit
   dashboard (Open positions tab) or run `python -m trading_bot.main approve <id>`.
5. Approved trades open in the paper portfolio. Stop loss / take profit /
   trailing stop are enforced on each cycle (the orchestrator calls
   `broker.update_open_positions()` after every scan).
6. Every yes-and-no decision is written to `decision_logs` — you can audit
   why a trade was or wasn't recommended.

### Tightening it further

- Set `HUMAN_APPROVAL_MODE=true` (default) — already the case.
- Bump `min_score_to_trade` and `min_credibility_to_trade` in `config.py`.
- Add symbols to `blacklist` in `config.py`.
- Lower `max_position_pct` and `max_daily_loss_pct`.

### What would it take to enable live trading?

By design, **a lot**. You would need to:

1. Write `trading/alpaca_broker.py` (or similar) subclassing `BrokerBase`,
   and have it refuse to place orders unless `LIVE_TRADING=true` AND a
   confirmation token is supplied.
2. Wire it into `TradingBotApp.__init__` based on `settings.broker`.
3. Set `LIVE_TRADING=true` in `.env`.
4. Set `HUMAN_APPROVAL_MODE=true` so every order still requires you to
   click approve in the dashboard.

The MVP intentionally stops at step 0.

## Design principles

- This is not financial advice.
- Every recommendation is explained (`reasoning` field).
- The system prefers `watch` over `buy` when confidence is low.
- High-impact, politically charged posts default to **watch-only** unless
  corroborated by an independent source.
- The system logs **why it did not trade**, not just why it did.

## License

MIT — for educational/demo use only. Use at your own risk.
