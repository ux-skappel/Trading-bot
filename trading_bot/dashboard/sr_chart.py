"""Candlestick chart with support/resistance levels (Altair)."""

from __future__ import annotations

from typing import List

import altair as alt
import pandas as pd

from ..analysis.support_resistance import Bar, SupportResistance

# Candles are context, so they stay neutral (hollow = up, filled = down);
# the levels carry color. Blue/red is a CVD-safe pair on light and dark.
CANDLE = "#898781"
SUPPORT = "#2a78d6"
RESISTANCE = "#e34948"
LABEL_INK = {"light": "#52514e", "dark": "#c3c2b7"}

LEVEL_COLUMNS = ["kind", "price", "distance_pct", "touches", "strength", "last_touch", "label"]


def _fmt(price: float) -> str:
    return f"{price:,.2f}" if price >= 1 else f"{price:.4g}"


def bars_frame(bars: List[Bar]) -> pd.DataFrame:
    df = pd.DataFrame([
        {"timestamp": b.timestamp, "open": b.open, "high": b.high,
         "low": b.low, "close": b.close, "volume": b.volume}
        for b in bars
    ])
    df["body_top"] = df[["open", "close"]].max(axis=1)
    df["body_bottom"] = df[["open", "close"]].min(axis=1)
    df["direction"] = (df["close"] >= df["open"]).map({True: "up", False: "down"})
    return df


def levels_frame(sr: SupportResistance) -> pd.DataFrame:
    """Levels as a price ladder: highest resistance first, lowest support last."""
    rows = [
        {
            "kind": lv.kind.capitalize(),
            "price": round(lv.price, 2 if lv.price >= 1 else 6),
            "distance_pct": round(lv.distance_pct * 100, 2),
            "touches": lv.touches,
            "strength": lv.strength,
            "last_touch": lv.last_touch,
            "label": f"{'R' if lv.kind == 'resistance' else 'S'} {_fmt(lv.price)} · {lv.touches}x",
        }
        for lv in sorted(sr.resistances, key=lambda lv: lv.price, reverse=True) + sr.supports
    ]
    return pd.DataFrame(rows, columns=LEVEL_COLUMNS)


def _labelled(levels: pd.DataFrame, y_range: float) -> pd.DataFrame:
    """Drop labels that would collide with a stronger level's label."""
    min_gap = 0.04 * y_range
    kept: List[int] = []
    for idx in levels.sort_values("strength", ascending=False).index:
        price, kind = levels.at[idx, "price"], levels.at[idx, "kind"]
        if all(
            abs(price - levels.at[k, "price"]) >= min_gap or levels.at[k, "kind"] != kind
            for k in kept
        ):
            kept.append(idx)
    return levels.loc[sorted(kept)]


def sr_chart(
    bars: List[Bar], sr: SupportResistance, height: int = 420, theme: str = "light"
) -> alt.LayerChart:
    """Candles plus one dashed, labelled line per support/resistance level.

    Resistance labels sit above their line and support labels below, so a
    support just under a resistance doesn't overprint it.
    """
    df = bars_frame(bars)
    levels = levels_frame(sr)
    bar_size = max(1, min(10, 600 // max(1, len(df))))
    ink = LABEL_INK.get(theme, LABEL_INK["light"])
    prices = list(df["high"]) + list(df["low"]) + list(levels["price"])
    shown = _labelled(levels, max(prices) - min(prices))

    span_days = (df["timestamp"].max() - df["timestamp"].min()).days
    date_format = "%b %Y" if span_days > 300 else "%b %d"
    base = alt.Chart(df).encode(
        x=alt.X("timestamp:T", title=None, axis=alt.Axis(format=date_format))
    )
    y = alt.Y("high:Q", title=None, scale=alt.Scale(zero=False, nice=False, padding=12))

    upper_wick = base.mark_rule(color=CANDLE).encode(y=y, y2="body_top:Q")
    lower_wick = base.mark_rule(color=CANDLE).encode(y="body_bottom:Q", y2="low:Q")
    body = base.mark_bar(size=bar_size, stroke=CANDLE, strokeWidth=1).encode(
        y="body_bottom:Q",
        y2="body_top:Q",
        fill=alt.condition("datum.direction === 'up'", alt.value("transparent"), alt.value(CANDLE)),
    )

    hover = alt.selection_point(
        fields=["timestamp"], nearest=True, on="mouseover", empty=False, clear="mouseout"
    )
    crosshair = base.mark_rule(color=CANDLE, strokeWidth=1).encode(
        opacity=alt.condition(hover, alt.value(0.6), alt.value(0.0)),
        tooltip=[
            alt.Tooltip("timestamp:T", title="Date"),
            alt.Tooltip("open:Q", title="Open", format=",.4~f"),
            alt.Tooltip("high:Q", title="High", format=",.4~f"),
            alt.Tooltip("low:Q", title="Low", format=",.4~f"),
            alt.Tooltip("close:Q", title="Close", format=",.4~f"),
            alt.Tooltip("volume:Q", title="Volume", format=",.0f"),
        ],
    ).add_params(hover)

    color = alt.Color(
        "kind:N",
        scale=alt.Scale(domain=["Support", "Resistance"], range=[SUPPORT, RESISTANCE]),
        legend=alt.Legend(title=None, orient="top"),
    )
    level_tooltip = [
        alt.Tooltip("kind:N", title="Level"),
        alt.Tooltip("price:Q", title="Price", format=",.4~f"),
        alt.Tooltip("distance_pct:Q", title="Distance %", format="+.2f"),
        alt.Tooltip("touches:Q", title="Touches"),
        alt.Tooltip("strength:Q", title="Strength", format=".2f"),
        alt.Tooltip("last_touch:T", title="Last touch"),
    ]
    level_base = alt.Chart(levels).encode(y="price:Q")
    lines = level_base.mark_rule(strokeWidth=2, strokeDash=[6, 4]).encode(color=color)
    label_base = alt.Chart(shown).encode(y="price:Q", x=alt.value(0), text="label:N")
    resistance_labels = label_base.mark_text(
        align="left", baseline="bottom", dx=4, dy=-3, color=ink
    ).transform_filter("datum.kind === 'Resistance'")
    support_labels = label_base.mark_text(
        align="left", baseline="top", dx=4, dy=3, color=ink
    ).transform_filter("datum.kind === 'Support'")
    # Invisible wide rule so level tooltips don't need pixel-perfect hovering.
    level_hit = level_base.mark_rule(strokeWidth=12, opacity=0.001).encode(tooltip=level_tooltip)

    return alt.layer(
        upper_wick, lower_wick, body, lines, resistance_labels, support_labels,
        crosshair, level_hit,
    ).properties(height=height)
