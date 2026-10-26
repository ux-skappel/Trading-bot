"""Charts for the support-bounce backtest (Altair)."""

from __future__ import annotations

from typing import List

import altair as alt
import pandas as pd

from ..analysis.support_resistance import Bar
from ..backtesting.support_bounce import BounceResult
from .sr_chart import CANDLE, LABEL_INK, RESISTANCE, SUPPORT

WIN, LOSS = SUPPORT, RESISTANCE          # same CVD-safe blue/red pair
SURFACE = {"light": "#ffffff", "dark": "#0e1117"}

TRADE_COLUMNS = [
    "entry_time", "entry_price", "exit_time", "exit_price", "return_pct",
    "exit_reason", "days", "support", "stop", "target", "outcome",
]


def trades_frame(result: BounceResult) -> pd.DataFrame:
    rows = [
        {
            "entry_time": t.entry_time,
            "entry_price": t.entry_price,
            "exit_time": t.exit_time,
            "exit_price": t.exit_price,
            "return_pct": round(t.return_pct * 100, 2),
            "exit_reason": t.exit_reason,
            "days": t.bars_held,
            "support": round(t.support, 6),
            "stop": round(t.stop, 6),
            "target": round(t.target, 6),
            "outcome": "Win" if t.return_pct > 0 else "Loss",
        }
        for t in result.trades
    ]
    return pd.DataFrame(rows, columns=TRADE_COLUMNS)


def trades_chart(
    bars: List[Bar], result: BounceResult, theme: str = "light", height: int = 340
) -> alt.LayerChart:
    """Price line with every trade drawn from entry to exit, colored by outcome."""
    start = result.equity[0][0] if result.equity else bars[0].timestamp
    prices = pd.DataFrame(
        [{"timestamp": b.timestamp, "close": b.close} for b in bars if b.timestamp >= start]
    )
    trades = trades_frame(result)

    price = alt.Chart(prices).mark_line(color=CANDLE, strokeWidth=1.5).encode(
        x=alt.X("timestamp:T", title=None),
        y=alt.Y("close:Q", title=None, scale=alt.Scale(zero=False)),
    )
    hover = alt.selection_point(
        fields=["timestamp"], nearest=True, on="mouseover", empty=False, clear="mouseout"
    )
    crosshair = alt.Chart(prices).mark_rule(color=CANDLE).encode(
        x="timestamp:T",
        opacity=alt.condition(hover, alt.value(0.6), alt.value(0.0)),
        tooltip=[alt.Tooltip("timestamp:T", title="Date"),
                 alt.Tooltip("close:Q", title="Close", format=",.4~f")],
    ).add_params(hover)

    color = alt.Color(
        "outcome:N",
        scale=alt.Scale(domain=["Win", "Loss"], range=[WIN, LOSS]),
        legend=alt.Legend(title=None, orient="top"),
    )
    tooltip = [
        alt.Tooltip("entry_time:T", title="Bought"),
        alt.Tooltip("entry_price:Q", title="Buy price", format=",.4~f"),
        alt.Tooltip("exit_time:T", title="Sold"),
        alt.Tooltip("exit_price:Q", title="Sell price", format=",.4~f"),
        alt.Tooltip("return_pct:Q", title="Return %", format="+.2f"),
        alt.Tooltip("exit_reason:N", title="Exit"),
        alt.Tooltip("support:Q", title="Support", format=",.4~f"),
    ]
    segments = alt.Chart(trades).mark_rule(strokeWidth=3).encode(
        x="entry_time:T", x2="exit_time:T", y="entry_price:Q", y2="exit_price:Q",
        color=color, tooltip=tooltip,
    )
    entries = alt.Chart(trades).mark_point(
        filled=True, size=80, opacity=1, stroke=SURFACE.get(theme, SURFACE["light"]), strokeWidth=2
    ).encode(x="entry_time:T", y="entry_price:Q", color=color, tooltip=tooltip)

    return alt.layer(price, crosshair, segments, entries).properties(height=height)


def equity_chart(result: BounceResult, theme: str = "light", height: int = 280) -> alt.LayerChart:
    """Value of 100 invested: strategy vs buy & hold, on one axis."""
    stamps = [ts for ts, _ in result.equity]
    wide = pd.DataFrame({
        "timestamp": stamps,
        "strategy": [v * 100 for _, v in result.equity],
        "buy_hold": [v * 100 for _, v in result.buy_hold],
    })
    long = pd.concat([
        pd.DataFrame({"timestamp": stamps, "series": "Strategy", "value": wide["strategy"]}),
        pd.DataFrame({"timestamp": stamps, "series": "Buy & hold", "value": wide["buy_hold"]}),
    ])
    color = alt.Color(
        "series:N",
        scale=alt.Scale(domain=["Strategy", "Buy & hold"], range=[SUPPORT, CANDLE]),
        legend=alt.Legend(title=None, orient="top"),
    )
    x = alt.X("timestamp:T", title=None)
    lines = alt.Chart(long).mark_line(strokeWidth=2).encode(
        x=x,
        y=alt.Y("value:Q", title="Value of 100 invested", scale=alt.Scale(zero=False)),
        color=color,
    )
    baseline = alt.Chart(pd.DataFrame({"value": [100]})).mark_rule(
        color=CANDLE, strokeWidth=1, opacity=0.5
    ).encode(y="value:Q")

    hover = alt.selection_point(
        fields=["timestamp"], nearest=True, on="mouseover", empty=False, clear="mouseout"
    )
    crosshair = alt.Chart(wide).mark_rule(color=CANDLE).encode(
        x="timestamp:T",
        opacity=alt.condition(hover, alt.value(0.6), alt.value(0.0)),
        tooltip=[
            alt.Tooltip("timestamp:T", title="Date"),
            alt.Tooltip("strategy:Q", title="Strategy", format=",.1f"),
            alt.Tooltip("buy_hold:Q", title="Buy & hold", format=",.1f"),
        ],
    ).add_params(hover)

    layers = [baseline, lines, crosshair]
    if not wide.empty:
        end = long.groupby("series").tail(1).copy()
        y_range = max(long["value"].max() - long["value"].min(), 1e-9)
        # Labels only when the line ends are far enough apart not to collide.
        if abs(end["value"].iloc[0] - end["value"].iloc[1]) > 0.06 * y_range:
            end["label"] = end["series"] + " " + (end["value"] - 100).map(lambda v: f"{v:+.0f}%")
            layers.append(
                alt.Chart(end).mark_text(
                    align="right", baseline="bottom", dx=-2, dy=-6,
                    color=LABEL_INK.get(theme, LABEL_INK["light"]),
                ).encode(x="timestamp:T", y="value:Q", text="label:N")
            )
    return alt.layer(*layers).properties(height=height)
