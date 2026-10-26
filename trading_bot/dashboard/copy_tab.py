"""Dashboard tab: paper-copy a trader's Telegram posts (read-only, no real orders)."""

from __future__ import annotations

import os
import tempfile

import altair as alt
import pandas as pd
import streamlit as st

from ..copytrade.engine import CopyConfig, CopyResult, replay
from ..copytrade.parser import parse_signal
from ..copytrade.prices import CandleFeed
from ..copytrade.store import export_authors, import_telegram_export, load_posts
from .sr_chart import SUPPORT

SETUP = """
**Two ways to get his posts in:**

1. **Upload a Telegram Desktop export** below: in Telegram Desktop open the group →
   ⋮ → *Export chat history* → untick photos/videos → format **JSON** → upload `result.json`.
2. **Telegram API** (also lets the bot follow new posts live): create an app at
   [my.telegram.org](https://my.telegram.org) → put `TELEGRAM_API_ID`, `TELEGRAM_API_HASH`,
   `COPY_CHAT` and `COPY_AUTHOR` in `.env`, then in Terminal:
   `python -m trading_bot.main copy login`, then `copy backfill --days 90`, then `copy listen`.
"""


@st.cache_data(ttl=300, show_spinner=False)
def _replay(posts, config: CopyConfig) -> CopyResult:
    return replay(posts, CandleFeed().candles, config)


def _parser_box() -> None:
    text = st.text_area("Test the parser: paste one of his posts", height=110,
                        placeholder="LONG $SOL entry 142-144 SL 138 TP1 150 TP2 158 10x")
    if text.strip():
        sig = parse_signal(text)
        (st.success if sig.tradable else st.info)(sig.summary())
        for note in sig.notes:
            st.caption(f"• {note}")


def _import_box() -> None:
    uploaded = st.file_uploader("Telegram Desktop export (result.json)", type=["json"], key="copy_export")
    if uploaded is None:
        return
    path = os.path.join(tempfile.gettempdir(), "telegram_result.json")
    with open(path, "wb") as f:
        f.write(uploaded.getvalue())
    try:
        senders = export_authors(path)
    except (ValueError, KeyError) as exc:
        st.error(f"That doesn't look like a Telegram JSON export: {exc}")
        return
    channel = "Every post is his (channel)"
    options = [f"{name} ({sender_id}) · {count} posts" for name, sender_id, count in senders[:30]]
    choice = st.selectbox("Whose posts should be copied?", options + [channel])
    if st.button("Import his posts", type="primary"):
        author = None if choice == channel else senders[options.index(choice)][1]
        counts = import_telegram_export(path, author)
        st.session_state["copy_imported"] = (
            f"Imported {counts['new']} new posts ({counts['skipped']} other messages skipped).")
        st.cache_data.clear()
        st.rerun()


def _equity_chart(result: CopyResult) -> alt.Chart:
    rows = [{"time": result.posts[0].post.time, "equity": result.start_equity}] if result.posts else []
    rows += [{"time": t, "equity": v} for t, v in result.equity]
    df = pd.DataFrame(rows)
    return alt.Chart(df).mark_line(strokeWidth=2, interpolate="step-after", color=SUPPORT).encode(
        x=alt.X("time:T", title=None),
        y=alt.Y("equity:Q", title="Paper account ($)", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("time:T", title="Time", format="%Y-%m-%d %H:%M"),
                 alt.Tooltip("equity:Q", title="Account", format="$,.2f")],
    ).properties(height=260)


def render(settings) -> None:
    st.subheader("Copy trading (paper)")
    st.caption(
        "Reads one trader's posts (read-only) and copies them as paper trades that follow his rules: "
        "his entry, stop, targets, leverage and size, plus his follow-up posts (close, move stop, cancel). "
        "Costs follow Liquid. No real orders are ever placed."
    )
    _parser_box()
    if "copy_imported" in st.session_state:
        st.success(st.session_state.pop("copy_imported"))

    posts = load_posts()
    with st.expander("Add his posts" if posts else "Get started: add his posts", expanded=not posts):
        st.markdown(SETUP)
        _import_box()
    if not posts:
        return

    config = CopyConfig.from_settings(settings)
    with st.spinner("Replaying his posts against real prices..."):
        result = _replay(posts, config)

    instructions = [o for o in result.posts if o.signal.action != "info"]
    open_trades = [t for t in result.trades if t.status == "open"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Posts read", len(result.posts))
    m2.metric("Trade instructions", len(instructions))
    m3.metric("Trades copied", len([t for t in result.trades if t.entry_price is not None]))
    m4.metric("Open now", len(open_trades))
    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Realized P/L", f"${result.realized_pnl:+,.0f}", f"{result.total_return:+.1%}")
    m6.metric("Win rate", "-" if result.win_rate is None else f"{result.win_rate:.0%}")
    m7.metric("Max drawdown", f"{result.max_drawdown:+.1%}")
    m8.metric("Entry vs his", "-" if result.avg_delay_cost is None else f"{result.avg_delay_cost:+.2%}",
              help="How much worse our average entry was than the entry he posted (positive = worse)")

    if result.equity:
        st.altair_chart(_equity_chart(result), use_container_width=True)

    st.markdown("#### Copied trades")
    st.dataframe(pd.DataFrame([{
        "#": t.number, "coin": t.symbol, "side": t.side, "leverage": t.leverage, "status": t.status,
        "posted": t.posted_at, "entry": t.entry_price, "his entry": t.his_entry,
        "entry vs his": None if t.delay_cost is None else t.delay_cost * 100,
        "exits": ", ".join(f"{f.reason} {f.price:g}" for f in t.fills),
        "P/L $": t.pnl + t.unrealized,
        "on margin": None if t.return_on_margin is None else t.return_on_margin * 100,
        "note": t.note,
    } for t in result.trades]), column_config={
        "leverage": st.column_config.NumberColumn(format="%gx"),
        "posted": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
        "entry vs his": st.column_config.NumberColumn(format="%+.2f%%"),
        "P/L $": st.column_config.NumberColumn(format="$%+.2f"),
        "on margin": st.column_config.NumberColumn(format="%+.1f%%"),
    }, hide_index=True, use_container_width=True)

    st.markdown("#### His posts and what the bot did")
    only_trades = st.toggle("Only trade instructions", value=True)
    rows = [{
        "time": o.post.time, "post": o.post.text[:160],
        "understood as": o.signal.summary(), "what happened": o.outcome,
    } for o in result.posts if not only_trades or o.signal.action != "info"]
    st.dataframe(pd.DataFrame(rows), column_config={
        "time": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
    }, hide_index=True, use_container_width=True, height=360)
    st.caption("Times are UTC. Paper only. Past results do not guarantee future returns.")
