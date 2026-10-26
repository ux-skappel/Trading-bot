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
- **Support & resistance** — every symbol the bot looks at gets automatic
  levels from 6 months of daily candles (clustered swing highs/lows, ranked
  by touches and recency). Stops go just below support, targets just below
  resistance, and a buy is downgraded to *watch* when reward/risk is below
  `min_reward_risk` (default 1.5). Shown as a chart per symbol in the
  dashboard; you can also upload your own CSV (Nordnet-style `;` and decimal
  commas work).
- **Strategy test: bounce off support** — backtests buying when a daily
  candle dips into a support zone and closes green above it (stop below the
  support, target below the nearest resistance, minimum reward/risk) over up
  to 10 years of real prices. Uses the same level detector as the live bot
  with only the data known on each day. Short-term exits: target below
  resistance, an optional trailing stop (break-even at +1R, then 1 ATR below
  the high) and a max number of days per trade. Position size can be set by
  risk per trade (a stop-out costs e.g. 1% of the account) with a leverage
  cap, or by fixed leverage (backtest only). Costs follow Liquid perps (0.095% fee per
  side, 0.03% funding per day, liquidation at 1.25% maintenance margin).
  Reports win rate, return vs buy & hold, max drawdown, liquidations and
  every trade.
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
    support_resistance.py # automatic S/R levels + stop/target from levels
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
    sr_chart.py           # candlestick chart with S/R lines
  copytrade/              # paper-copy a trader's Telegram posts
    parser.py             # post -> trade instruction
    engine.py             # replay posts as paper trades (his rules)
    prices.py             # Hyperliquid/Yahoo intraday candles
    telegram_reader.py    # read-only Telegram (Telethon)
    store.py              # saved posts + Telegram Desktop import
  backtesting/
    backtest.py           # news-event replay
    support_bounce.py     # support-bounce strategy backtest
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

7. **Support/resistance levels in the terminal:**
   ```bash
   python -m trading_bot.main levels AAPL EQNR.OL BTC   # chosen symbols
   python -m trading_bot.main levels                    # whole watchlist
   python -m trading_bot.main levels --csv my_chart.csv # your own data
   ```
   Or open the **Support/Resistance** tab in the dashboard.

   **Test the support-bounce strategy** (or use the **Strategy test** tab):
   ```bash
   python -m trading_bot.main strategy --market us crypto       # 5 years
   python -m trading_bot.main strategy AAPL BTC --period 10y --trades
   python -m trading_bot.main strategy --min-touches 3 --min-rr 2
   python -m trading_bot.main strategy --market crypto --leverage 5 --max-days 3
   python -m trading_bot.main strategy --market us crypto --risk 1 --leverage 5 --trailing --max-days 3
   ```

8. **Run the tests:**
   ```bash
   pytest -q
   ```

9. **Backtest with the bundled sample data:**
   ```bash
   python -m trading_bot.main backtest \
     --events trading_bot/data/sample_events.csv \
     --prices trading_bot/data/sample_prices.csv \
     --horizon 5
   ```

## Copy trading (paper, read-only)

Copy a trader's Telegram posts as **paper trades that follow his rules**: his
entry, stop loss, targets, leverage and size, plus his follow-up posts (close,
close part, move stop, cancel). Replays his past posts against real
Hyperliquid prices (the venue Liquid routes to), so you see his track record
and how much worse your entries are because you copy after he posts. Nothing
here sends Telegram messages or places real orders.

1. Get his posts in, either way:
   - **Telegram Desktop export:** open the group → ⋮ → *Export chat history* →
     JSON → upload `result.json` in the dashboard's **Copy trading** tab, or
     `python -m trading_bot.main copy import result.json --author "<his name>"`.
   - **Telegram API** (also follows new posts live): create an app at
     <https://my.telegram.org>, fill `TELEGRAM_API_ID`, `TELEGRAM_API_HASH`,
     `TELEGRAM_PHONE`, `COPY_CHAT`, `COPY_AUTHOR` (and `COPY_TOPIC` for a group with topics) in `.env`, then:
     ```bash
     python -m trading_bot.main copy login          # once; asks for the Telegram code
     python -m trading_bot.main copy chats          # find the group
     python -m trading_bot.main copy authors        # find his id/username
     python -m trading_bot.main copy topics         # group with topics: find his trade topic
     python -m trading_bot.main copy backfill --days 90
     python -m trading_bot.main copy listen         # keep running to follow new posts
     ```
     Your login is stored in `telegram.session`: keep it private (git-ignored).
2. See the result in the **Copy trading** tab, or `python -m trading_bot.main copy report`.
3. Check how a post is understood: paste it in the tab's parser box, or
   `python -m trading_bot.main copy parse "LONG $SOL entry 142 SL 138 TP 150 10x"`.

**Live paper account vs. test on old posts.** The live account starts the
first time you run `copy listen` (restart it with `copy reset-live`): only his
posts after that are bought, at the price captured the moment the bot sees
them, and a buy seen more than `COPY_MAX_OPEN_DELAY_MINUTES` (default 2) after
he posted is skipped. Sells and stop moves on positions you hold still happen
when seen. `copy report` shows the live account; `copy report --history` (or
the toggle in the tab) replays all stored posts as a clearly labelled test.
`copy review` writes `review.txt` (git-ignored, stays on your computer).

**Images.** When he sells with a picture of the trade and a caption like
"SELGER HELE NÅ", the bot reads the text in the picture with the Mac's
built-in text recognition (offline, free, no API key). Two kinds of picture
are trusted: his **Liquid trade card** (coin, side, leverage, entry and mark
price) and a **price chart** (only the coin in its title). A card picks the
exact position: a sell whose card shows his 75x long is not applied to the
10x short the bot copied, and a sell of "my 100x" keeps a 40x copy.
In the discussion topics only cards count. News screenshots, SizeProp cards
and memes say nothing. A picture never opens a trade by itself, but a buy
post without leverage takes the leverage from his card. Pictures are read
only on sells, stop moves and buys without leverage (the rest are skipped to
save time); replies to images are still followed back to the original trade
post. `copy images` shows what was read.

**Not copied:** status updates ("fortsatt long", "for øyeblikket
long"), plans ("selger ved $X", "tror jeg selger i kveld", "går inn
kl 21:58"), anything about his SizeProp/prop accounts, candle talk
("1h close"), memes, and chat that mentions long/short without a call.
**Copied too:** calls written out in words (a company named in the post plus
"går long med 10X giring"), several calls in one sentence ("long A og B +
short C"), adds ("longet mer", which take the coin and
leverage from the post he replied to), a stop move per coin line, and a sell
plus a stop move in one post. `copy review` writes every post with what the
bot understood and did, to check for misreads.

**Sells posted in other topics.** `COPY_SELL_TOPICS` (e.g. his stocks and
crypto discussion topics) are read too, but only for sells and stop updates,
never buys, and a sell there that doesn't say what was sold (no coin, reply
or picture) is ignored rather than guessed. The Copy trading tab lists
positions open for more than 3 days without a sell, with whether they have a
stop loss.

**Size by how sure he sounds.** When his post gives no size, the default
margin is scaled by his wording: "veldig risky", "risky ⚠️⚠️" or "gamble"
→ ×0.25; "risky", ⚠️ or "ikke stor posisjon" → ×0.5; "klink support",
"ser mega bra ut", "liker risk-rewarden" or "oversolgt og på support" → ×1.5.
Risky wins when a post says both. Change the multipliers with
`COPY_SIZE_VERY_RISKY`, `COPY_SIZE_RISKY` and `COPY_SIZE_STRONG`.

Rules when his post leaves something open: enter at once if the price is
within 1% of his entry, otherwise wait at his entry for 24 h; size by his risk
% (with his stop), else his size %, else 10% of the account as margin scaled
by how sure he sounds; split
the position equally across his targets; stop-first when a candle touches both.
Posts that aren't clear trade instructions are never traded and are listed
with the reason.

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
