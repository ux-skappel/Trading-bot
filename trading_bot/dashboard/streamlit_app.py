"""Streamlit dashboard.

Run with:
    streamlit run trading_bot/dashboard/streamlit_app.py
"""

from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import streamlit as st
from sqlalchemy import select

from trading_bot.app import TradingBotApp
from trading_bot.config import get_settings
from trading_bot.storage.database import init_db, session_scope
from trading_bot.storage.models import (...)
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
tab_news, tab_opps, tab_portfolio, tab_positions, tab_perf, tab_logs, tab_sources, tab_watch, tab_config = (
    st.tabs([
        "News", "Opportunities", "Portfolio", "Open positions", "Performance",
        "Decisions", "Sources", "Watchlists", "Config",
    ])
)

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
    state = port.snapshot()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cash", f"${state.cash:,.2f}")
    c2.metric("Positions value", f"${state.positions_value:,.2f}")
    c3.metric("Total value", f"${state.total_value:,.2f}")
    c4.metric("Daily P/L", f"${state.daily_pnl:,.2f}")
    with session_scope() as db:
        snaps = list(
            db.scalars(
                select(PortfolioSnapshot).order_by(
                    PortfolioSnapshot.timestamp.asc()
                ).limit(500)
            )
        )
    if snaps:
        st.line_chart(
            pd.DataFrame(
                {"timestamp": [s.timestamp for s in snaps],
                 "total_value": [s.total_value for s in snaps]}
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
