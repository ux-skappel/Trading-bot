"""Streamlit dashboard.

Run with:
    streamlit run trading_bot/dashboard/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import streamlit as st
from sqlalchemy import select

from trading_bot.analysis.support_resistance import bars_from_csv_text, detect_levels
from trading_bot.app import TradingBotApp
from trading_bot.backtesting.support_bounce import BounceParams, backtest_support_bounce, summarize
from trading_bot.config import get_settings
from trading_bot.connectors.market_data import MarketDataProvider, infer_asset_class
from trading_bot.dashboard.backtest_chart import (
    EXIT_LABEL,
    equity_chart,
    trade_chart,
    trade_summary,
    trades_chart,
    trades_frame,
)
from trading_bot.dashboard import copy_tab
from trading_bot.dashboard.sr_chart import levels_frame, sr_chart
from trading_bot.storage.database import init_db, session_scope
from trading_bot.storage.models import (
    DecisionLog,
    HighImpactAccount,
    NormalizedEvent,
    OpportunityScore,
    PaperTrade,
    PortfolioSnapshot,
    Source,
)
from trading_bot.trading.paper_broker import PaperBroker
from trading_bot.trading.portfolio import Portfolio


st.set_page_config(page_title="Trading Bot — MVP", layout="wide")
init_db()
settings = get_settings()


def _df(rows) -> pd.DataFrame:
    """Convert SQLAlchemy rows to a DataFrame for display."""
    out = []
    for r in rows:
        d = {c.name: getattr(r, c.name) for c in r.__table__.columns}
        out.append(d)
    return pd.DataFrame(out)


@st.cache_data(ttl=300, show_spinner=False)
def _levels_for_symbol(symbol: str):
    """Fetch daily history for a symbol and detect its levels (cached 5 min)."""
    history = MarketDataProvider(settings.market_data_provider).get_history(
        symbol, infer_asset_class(symbol)
    )
    sr = detect_levels(history.bars, symbol=symbol,
                       window=settings.sr_window, max_levels=settings.sr_max_levels)
    return history.bars, sr, history.source


@st.cache_data(ttl=3600, show_spinner=False)
def _bounce_for_symbol(symbol: str, period: str, min_touches: int, min_rr: float,
                       leverage: int, max_days, risk, trailing: bool):
    """Backtest the support-bounce strategy for one symbol (cached 1 hour)."""
    history = MarketDataProvider(settings.market_data_provider).get_history(
        symbol, infer_asset_class(symbol), period=period
    )
    params = BounceParams.from_settings(
        settings, min_touches=min_touches, min_reward_risk=min_rr,
        leverage=leverage, max_hold_days=max_days, risk_per_trade=risk, trailing=trailing,
    )
    return history.bars, backtest_support_bounce(history.bars, symbol, params), history.source


LEVEL_TABLE_CONFIG = {
    "distance_pct": st.column_config.NumberColumn("distance", format="%+.2f%%"),
    "last_touch": st.column_config.DateColumn("last touch", format="YYYY-MM-DD"),
    "label": None,
}


def _theme() -> str:
    """Active light/dark theme (older Streamlit versions don't expose it)."""
    theme = getattr(getattr(st, "context", None), "theme", None)
    return getattr(theme, "type", None) or "light"


def _show_levels(bars, sr, caption: str) -> None:
    if len(bars) < 2:
        st.warning("Not enough price data to find levels.")
        return
    st.altair_chart(sr_chart(bars, sr, theme=_theme()), use_container_width=True)
    st.caption(f"{sr.summary()} · {caption} · tolerance {sr.tolerance_pct:.1%}")
    st.dataframe(levels_frame(sr), column_config=LEVEL_TABLE_CONFIG,
                 hide_index=True, use_container_width=True)


st.title("Trading Bot — MVP (paper trading)")

# Safety banner
banner = (
    f"**Mode:** `{settings.trading_mode}`  ·  "
    f"**Live trading:** `{settings.live_trading}`  ·  "
    f"**Human approval:** `{settings.human_approval_mode}`  ·  "
    f"**Broker:** `{settings.broker}`"
)
if settings.live_trading and settings.broker != "paper":
    st.error(banner + "  — ⚠️ live broker code path NOT shipped in this MVP")
else:
    st.success(banner + "  — paper trading only. This is **not** financial advice.")

# Manual scan trigger
col_btn, col_msg = st.columns([1, 4])
with col_btn:
    if st.button("Run scan now"):
        with st.spinner("Running pipeline..."):
            res = TradingBotApp().run_once()
        st.session_state["last_result"] = (
            f"ingested={res.ingested} normalized={res.normalized} "
            f"opportunities={res.opportunities} high_impact={res.high_impact_hits}"
        )
with col_msg:
    st.write(st.session_state.get("last_result", "Press 'Run scan now' to ingest fresh data."))

# Tabs
(tab_news, tab_opps, tab_levels, tab_strategy, tab_copy, tab_portfolio, tab_positions, tab_perf,
 tab_logs, tab_sources, tab_watch, tab_config) = st.tabs([
    "News", "Opportunities", "Support/Resistance", "Strategy test", "Copy trading", "Portfolio",
    "Open positions", "Performance", "Decisions", "Sources", "Watchlists", "Config",
])

with tab_copy:
    copy_tab.render(settings)

with tab_strategy:
    st.subheader("Strategy test: bounce off support")
    st.caption(
        "Buys when a daily candle dips into a support zone, holds, and closes green above it. "
        "Enters at the next open if reward/risk still holds; stop just below the support, "
        "target just below the nearest resistance. Uses the same level detector as the live "
        "bot, with only the data known on each day. Costs follow Liquid perps: 0.095% fee per "
        "side, 0.03% funding per day and liquidation at 1.25% maintenance margin. Leverage "
        "applies to this test only. Past results do not guarantee future returns."
    )
    market_keys = {"US stocks": "us", "Oslo Børs": "oslo", "Europe": "europe", "Crypto": "crypto"}
    hold_options = {"No limit": None, "1 day": 1, "2 days": 2, "3 days": 3, "5 days": 5, "10 days": 10}
    risk_options = {"Off (fixed leverage)": None, "0.5%": 0.005, "1%": 0.01, "2%": 0.02, "5%": 0.05}
    c_mkt, c_per, c_hold, c_trail = st.columns([3, 1, 1, 1])
    markets = c_mkt.multiselect("Markets", list(market_keys), default=["US stocks", "Crypto"])
    period = c_per.selectbox("History", ["1y", "2y", "5y", "10y"], index=2)
    hold_label = c_hold.selectbox("Max days in trade", list(hold_options), index=3)
    trailing = c_trail.toggle(
        "Trailing stop", value=True,
        help="Once price has risen as much as you risked, the stop moves to break-even and then "
             "follows 1 ATR (a normal day's range) below the highest price, so a bounce that "
             "turns back is sold before it falls again.",
    )
    c_risk, c_lev, c_tch, c_rr = st.columns(4)
    risk_label = c_risk.selectbox(
        "Risk per trade", list(risk_options), index=2,
        help="How much of the account a trade loses if the stop is hit. The position is sized "
             "to match, so a tight stop uses more leverage and a wide stop less.",
    )
    leverage = c_lev.selectbox(
        "Max leverage" if risk_options[risk_label] else "Leverage",
        [1, 2, 3, 5, 10, 20, 40], index=3, format_func=lambda x: f"{x}x",
        help="Liquid allows up to 40x on crypto. With risk per trade on, this is the cap.",
    )
    min_touches = int(c_tch.number_input("Min touches", min_value=1, max_value=5, value=2))
    min_rr = float(c_rr.number_input("Min reward/risk", min_value=1.0, max_value=5.0,
                                     value=float(settings.min_reward_risk), step=0.25))
    if st.button("Run backtest", type="primary", disabled=not markets):
        st.session_state["bounce_request"] = (
            tuple(settings.symbols_for([market_keys[m] for m in markets])),
            period, min_touches, min_rr, leverage, hold_options[hold_label],
            risk_options[risk_label], trailing,
        )

    request = st.session_state.get("bounce_request")
    if request is None:
        st.info("Pick markets and press **Run backtest**.")
    else:
        symbols, req_period, req_touches, req_rr, req_lev, req_days, req_risk, req_trail = request
        runs = {}
        progress = st.progress(0.0, text="Backtesting...")
        for i, sym in enumerate(symbols, start=1):
            runs[sym] = _bounce_for_symbol(sym, req_period, req_touches, req_rr, req_lev,
                                           req_days, req_risk, req_trail)
            progress.progress(i / len(symbols), text=f"Backtesting {sym}...")
        progress.empty()

        results = [res for _, res, _ in runs.values()]
        offline = [sym for sym, (_, _, src) in runs.items() if src != "yfinance"]
        if offline:
            st.warning(f"No live price data for {', '.join(offline)} — those use offline demo data.")
        sizing = (f"risk {req_risk:.1%} per trade, max {req_lev}x" if req_risk
                  else f"{req_lev}x leverage")
        st.caption(
            f"{len(symbols)} symbols · {req_period} of daily candles · {sizing} · "
            f"{'trailing stop' if req_trail else 'fixed stop'} · "
            f"{'max ' + str(req_days) + ' days in trade' if req_days else 'no time limit'} · "
            f"min {req_touches} touches · min reward/risk {req_rr}"
        )

        total = summarize(results)
        a1, a2 = st.columns(2)
        a1.metric("Whole account", f"{total['account_return']:+.1%}",
                  help="One account taking every trade of every symbol, each sized on the whole "
                       "account (e.g. risking 5% of it). This is what following all signals does.")
        a2.metric("Whole account: biggest fall", f"{total['account_max_drawdown']:+.1%}",
                  help="Largest drop from a high point of that one account")
        m1, m2, m3, m4, m4b = st.columns(5)
        m1.metric("Trades", total["trades"])
        m2.metric("Win rate", "-" if total["win_rate"] is None else f"{total['win_rate']:.0%}")
        m3.metric("Avg per trade", "-" if total["avg_return"] is None else f"{total['avg_return']:+.2%}",
                  help="Return on your own money, after leverage, fees and funding")
        m4.metric("Worst trade", "-" if total["worst_trade"] is None else f"{total['worst_trade']:+.1%}",
                  help="Can be worse than the risk per trade when price gaps past the stop")
        m4b.metric("Profit factor", "-" if total["profit_factor"] is None else f"{total['profit_factor']:.2f}",
                   help="Total gains divided by total losses. Below 1 loses money over time.")
        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Beat buy & hold", f"{total['beat_buy_hold']} of {total['symbols']}")
        m6.metric("Avg days held", "-" if total["avg_days_held"] is None else f"{total['avg_days_held']:.1f}")
        m7.metric("Liquidations", total["liquidations"])
        m8.metric("Accounts wiped out", f"{total['wiped_out']} of {total['symbols']}")

        def _pct100(value):
            return None if value is None else value * 100

        overview = pd.DataFrame([
            {
                "symbol": r.symbol,
                "trades": len(r.trades),
                "win rate": _pct100(r.win_rate),
                "avg per trade": _pct100(r.avg_return),
                "strategy": r.total_return * 100,
                "buy & hold": r.buy_hold_return * 100,
                "max drawdown": r.max_drawdown * 100,
                "time in market": r.time_in_market * 100,
                "liquidations": r.liquidations,
            }
            for r in results
        ]).sort_values("strategy", ascending=False)
        pct = st.column_config.NumberColumn(format="%.1f%%")
        signed = st.column_config.NumberColumn(format="%+.1f%%")
        st.dataframe(
            overview,
            column_config={
                "win rate": pct, "time in market": pct, "avg per trade": signed,
                "strategy": signed, "buy & hold": signed, "max drawdown": signed,
            },
            hide_index=True,
            use_container_width=True,
        )

        trade_table_config = {
            "return_pct": st.column_config.NumberColumn("return", format="%+.2f%%"),
            "entry_time": st.column_config.DateColumn("bought", format="YYYY-MM-DD"),
            "entry_price": "buy price",
            "exit_time": st.column_config.DateColumn("sold", format="YYYY-MM-DD"),
            "exit_price": "sell price",
            "exit_reason": "exit",
            "days": "days held",
            "size": st.column_config.NumberColumn(
                "size", format="%.2fx", help="Position as a multiple of the account"),
            "outcome": None,
        }

        st.markdown("#### All buys and sells")
        all_trades = pd.concat([trades_frame(r) for r in results]).sort_values("entry_time")
        st.dataframe(all_trades, column_config=trade_table_config, hide_index=True,
                     use_container_width=True, height=360)
        st.download_button(
            "Download all trades (CSV for Excel)",
            all_trades.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
            file_name="all_trades.csv",
            mime="text/csv",
        )

        st.markdown("#### One symbol")
        pick = st.selectbox("Show trades for", list(overview["symbol"]))
        pick_bars, pick_result, _ = runs[pick]
        st.altair_chart(trades_chart(pick_bars, pick_result, theme=_theme()),
                        use_container_width=True)
        st.altair_chart(equity_chart(pick_result, theme=_theme()), use_container_width=True)
        st.dataframe(trades_frame(pick_result), column_config=trade_table_config,
                     hide_index=True, use_container_width=True)

        if pick_result.trades:
            st.markdown("#### One trade up close")
            labels = [
                f"{t.entry_time:%Y-%m-%d} · {t.return_pct:+.2%} · {EXIT_LABEL[t.exit_reason]}"
                for t in pick_result.trades
            ]
            chosen = st.selectbox("Trade", range(len(labels)), format_func=lambda i: labels[i])
            trade = pick_result.trades[chosen]
            st.markdown(trade_summary(pick_bars, trade))
            st.altair_chart(trade_chart(pick_bars, trade, theme=_theme()), use_container_width=True)
            st.caption(
                "30 days before the buy to 15 days after the sell. Shaded: days in the trade. "
                "Dashed lines: every support and resistance the bot saw the day before it bought. "
                "Solid lines: the stop loss and target for this trade."
            )

with tab_levels:
    st.subheader("Support & resistance")
    st.caption(
        "Levels are clustered swing highs/lows from daily candles, ranked by touches "
        "and recency. The bot places stops just below support and targets just below "
        "resistance, and downgrades a buy to watch when reward/risk is too low."
    )
    with session_scope() as db:
        opp_symbols = sorted(db.scalars(select(OpportunityScore.symbol).distinct()))
    all_symbols = list(dict.fromkeys(
        settings.stock_watchlist + settings.crypto_watchlist + opp_symbols
    ))

    choice = st.selectbox("Symbol", all_symbols)
    with st.spinner(f"Loading {choice}..."):
        bars, sr, source = _levels_for_symbol(choice)
    _show_levels(bars, sr, f"data: {source}, {len(bars)} daily bars")

    st.divider()
    if st.toggle(f"Show levels for all {len(all_symbols)} symbols"):
        def _px(value):
            return None if value is None else round(value, 2 if value >= 1 else 6)

        overview = []
        progress = st.progress(0.0)
        for i, sym in enumerate(all_symbols, start=1):
            _, s_r, src = _levels_for_symbol(sym)
            sup, res = s_r.nearest_support, s_r.nearest_resistance
            overview.append({
                "symbol": sym,
                "price": _px(s_r.current_price),
                "support": _px(sup.price) if sup else None,
                "to support": sup.distance_pct * 100 if sup else None,
                "resistance": _px(res.price) if res else None,
                "to resistance": res.distance_pct * 100 if res else None,
                "data": src,
            })
            progress.progress(i / len(all_symbols))
        progress.empty()
        st.dataframe(
            pd.DataFrame(overview),
            column_config={
                "to support": st.column_config.NumberColumn(format="%+.2f%%"),
                "to resistance": st.column_config.NumberColumn(format="%+.2f%%"),
            },
            hide_index=True,
            use_container_width=True,
        )

    st.divider()
    uploaded = st.file_uploader(
        "Your own chart data (CSV with date + close, optionally open/high/low/volume)",
        type=["csv", "txt"],
    )
    if uploaded is not None:
        try:
            up_bars = bars_from_csv_text(uploaded.getvalue().decode("utf-8-sig"))
        except (ValueError, UnicodeDecodeError) as exc:
            st.error(f"Could not read {uploaded.name}: {exc}")
        else:
            up_sr = detect_levels(up_bars, symbol=uploaded.name,
                                  window=settings.sr_window, max_levels=settings.sr_max_levels)
            _show_levels(up_bars, up_sr, f"{len(up_bars)} rows from {uploaded.name}")

with tab_news:
    st.subheader("Latest detected news (normalized)")
    with session_scope() as db:
        rows = list(
            db.scalars(select(NormalizedEvent).order_by(NormalizedEvent.timestamp.desc()).limit(100))
        )
        for r in rows:
            db.expunge(r)
    if rows:
        df = pd.DataFrame([
            {
                "time": r.timestamp,
                "source": r.source,
                "author": r.author,
                "tickers": r.tickers,
                "crypto": r.crypto_symbols,
                "event": r.event_type,
                "sentiment": r.sentiment,
                "high_impact": r.is_high_impact_account,
                "topics": r.high_impact_topics,
                "rumor": r.is_rumor,
                "url": r.url,
                "text": (r.text or "")[:200],
            }
            for r in rows
        ])
        st.dataframe(df, use_container_width=True, height=600)
    else:
        st.info("No normalized events yet. Run a scan.")

with tab_opps:
    st.subheader("Ranked opportunities")
    with session_scope() as db:
        rows = list(
            db.scalars(
                select(OpportunityScore).order_by(
                    OpportunityScore.timestamp.desc()
                ).limit(200)
            )
        )
        for r in rows:
            db.expunge(r)
    if rows:
        df = pd.DataFrame([
            {
                "time": r.timestamp,
                "symbol": r.symbol,
                "class": r.asset_class,
                "direction": r.direction,
                "score": r.total_score,
                "confidence": r.confidence,
                "risk": r.risk_level,
                "reason": (r.reasoning or "")[:300],
            }
            for r in rows
        ])
        df = df.sort_values("score", ascending=False)
        st.dataframe(df, use_container_width=True, height=600)
    else:
        st.info("No scored opportunities yet.")

with tab_portfolio:
    st.subheader("Paper portfolio")
    port = Portfolio()
    state = port.snapshot(save=False)   # viewing the page shouldn't write to the database
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cash", f"${state.cash:,.2f}")
    c2.metric("Positions value", f"${state.positions_value:,.2f}")
    c3.metric("Total value", f"${state.total_value:,.2f}")
    c4.metric("Daily P/L", f"${state.daily_pnl:,.2f}")
    with session_scope() as db:
        snaps = [
            (s.timestamp, s.total_value)
            for s in db.scalars(
                select(PortfolioSnapshot).order_by(
                    PortfolioSnapshot.timestamp.asc()
                ).limit(500)
            )
        ]
    if snaps:
        st.line_chart(
            pd.DataFrame(
                {"timestamp": [t for t, _ in snaps],
                 "total_value": [v for _, v in snaps]}
            ).set_index("timestamp")
        )

with tab_positions:
    st.subheader("Open & pending positions — approve here")
    with session_scope() as db:
        opens = list(
            db.scalars(
                select(PaperTrade).where(
                    PaperTrade.status.in_(["open", "pending_approval"])
                ).order_by(PaperTrade.opened_at.desc())
            )
        )
        for r in opens:
            db.expunge(r)
    if not opens:
        st.info("No open or pending trades.")
    for t in opens:
        with st.expander(
            f"#{t.id} {t.status.upper()} {t.symbol} qty={t.quantity} "
            f"entry={t.entry_price:.4f}"
        ):
            st.write({
                "asset_class": t.asset_class,
                "side": t.side,
                "stop_loss": t.stop_loss,
                "take_profit": t.take_profit,
                "trailing_stop_pct": t.trailing_stop_pct,
                "reasoning": t.reasoning,
                "approved_by": t.approved_by,
            })
            if t.status == "pending_approval":
                col_a, col_b = st.columns(2)
                if col_a.button(f"Approve trade #{t.id}", key=f"appr-{t.id}"):
                    res = PaperBroker().approve_pending(t.id, approver="dashboard")
                    st.success(str(res))
                if col_b.button(f"Reject trade #{t.id}", key=f"rej-{t.id}"):
                    PaperBroker().cancel_order(t.id)
                    st.warning(f"trade #{t.id} cancelled")

with tab_perf:
    st.subheader("Historical performance")
    with session_scope() as db:
        closed = list(
            db.scalars(
                select(PaperTrade).where(PaperTrade.status == "closed")
                .order_by(PaperTrade.closed_at.desc())
            )
        )
        for r in closed:
            db.expunge(r)
    if not closed:
        st.info("No closed trades yet.")
    else:
        df = pd.DataFrame([
            {"closed_at": t.closed_at, "symbol": t.symbol, "side": t.side,
             "qty": t.quantity, "entry": t.entry_price, "exit": t.exit_price,
             "pnl": t.pnl, "exit_reason": t.exit_reason}
            for t in closed
        ])
        st.dataframe(df, use_container_width=True)
        wins = (df["pnl"] > 0).sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("Closed trades", len(df))
        c2.metric("Win rate", f"{(wins / max(1, len(df))) * 100:.1f}%")
        c3.metric("Total P/L", f"${df['pnl'].sum():,.2f}")

with tab_logs:
    st.subheader("Decision logs (every yes/no)")
    with session_scope() as db:
        logs = list(
            db.scalars(
                select(DecisionLog).order_by(DecisionLog.timestamp.desc()).limit(500)
            )
        )
        for r in logs:
            db.expunge(r)
    if logs:
        df = pd.DataFrame([
            {"time": l.timestamp, "action": l.action, "decision": l.decision,
             "symbol": l.symbol, "score": l.score, "reason": l.reason}
            for l in logs
        ])
        st.dataframe(df, use_container_width=True, height=500)
    else:
        st.info("No decisions logged yet.")

with tab_sources:
    st.subheader("Source credibility")
    with session_scope() as db:
        srcs = list(db.scalars(select(Source).order_by(Source.credibility.desc())))
        accounts = list(
            db.scalars(
                select(HighImpactAccount).order_by(HighImpactAccount.credibility.desc())
            )
        )
        for r in srcs:
            db.expunge(r)
        for r in accounts:
            db.expunge(r)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**News sources**")
        st.dataframe(_df(srcs), use_container_width=True)
    with col2:
        st.markdown("**High-impact accounts**")
        st.dataframe(_df(accounts), use_container_width=True)

with tab_watch:
    st.subheader("Watchlists")
    st.markdown("**Stocks**")
    st.write(settings.stock_watchlist)
    st.markdown("**Crypto**")
    st.write(settings.crypto_watchlist)
    st.markdown("**Blacklist**")
    st.write(settings.blacklist or "(empty)")

with tab_config:
    st.subheader("Runtime configuration")
    cfg = {
        "safety": settings.safety_summary(),
        "thresholds": {
            "min_score_to_trade": settings.min_score_to_trade,
            "min_score_to_recommend": settings.min_score_to_recommend,
            "min_credibility_to_trade": settings.min_credibility_to_trade,
        },
        "risk": {
            "starting_cash": settings.starting_cash,
            "max_position_pct": settings.max_position_pct,
            "max_daily_loss_pct": settings.max_daily_loss_pct,
            "max_open_positions": settings.max_open_positions,
            "max_exposure_per_class_pct": settings.max_exposure_per_class_pct,
            "cooldown_minutes_after_loss": settings.cooldown_minutes_after_loss,
            "extreme_volatility_pct": settings.extreme_volatility_pct,
        },
        "scoring_weights": settings.scoring_weights,
    }
    st.code(json.dumps(cfg, indent=2, default=str))
