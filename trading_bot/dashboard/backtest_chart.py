"""Charts for the support-bounce backtest (Altair)."""

from __future__ import annotations

from datetime import timedelta
from typing import List

import altair as alt
import pandas as pd

from ..analysis.support_resistance import Bar, SupportResistance
from ..backtesting.support_bounce import BounceResult, BounceTrade
from .sr_chart import CANDLE, LABEL_INK, RESISTANCE, SUPPORT, _fmt, sr_layers

WIN, LOSS = SUPPORT, RESISTANCE          # same CVD-safe blue/red pair
SURFACE = {"light": "#ffffff", "dark": "#0e1117"}

TRADE_COLUMNS = [
    "symbol", "entry_time", "entry_price", "exit_time", "exit_price", "return_pct",
    "exit_reason", "days", "size", "support", "stop", "target", "outcome",
]


def trades_frame(result: BounceResult) -> pd.DataFrame:
    rows = [
        {
            "symbol": t.symbol,
            "entry_time": t.entry_time,
            "entry_price": t.entry_price,
            "exit_time": t.exit_time,
            "exit_price": t.exit_price,
            "return_pct": round(t.return_pct * 100, 2),
            "exit_reason": t.exit_reason,
            "days": t.days_held,
            "size": round(t.position_size, 2),
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
        pd.DataFrame({"timestamp": stamps, "series": "Buy & hold (1x)", "value": wide["buy_hold"]}),
    ])
    color = alt.Color(
        "series:N",
        scale=alt.Scale(domain=["Strategy", "Buy & hold (1x)"], range=[SUPPORT, CANDLE]),
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
            alt.Tooltip("buy_hold:Q", title="Buy & hold (1x)", format=",.1f"),
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


EXIT_LABEL = {
    "stop": "stop loss", "trail": "trailing stop", "target": "target",
    "time": "time stop", "liquidated": "liquidated", "end_of_data": "still open",
}


def trade_window(bars: List[Bar], trade: BounceTrade, before: int = 30, after: int = 15) -> List[Bar]:
    """Candles from ``before`` days ahead of the buy to ``after`` days past the sell."""
    index = {b.timestamp: i for i, b in enumerate(bars)}
    start = max(0, index[trade.entry_time] - before)
    end = min(len(bars), index[trade.exit_time] + after + 1)
    return bars[start:end]


def trade_chart(
    bars: List[Bar], trade: BounceTrade, theme: str = "light", height: int = 440
) -> alt.LayerChart:
    """One trade in context: candles before, during and after it, every
    support/resistance level the bot saw on the signal day, the holding
    period, the stop and target, and the buy and sell."""
    window = trade_window(bars, trade)
    levels = trade.levels or SupportResistance(trade.symbol, trade.entry_price, 0.0)
    ink = LABEL_INK.get(theme, LABEL_INK["light"])
    surface = SURFACE.get(theme, SURFACE["light"])
    half_day = timedelta(hours=12)
    start, end = trade.entry_time - half_day, trade.exit_time + half_day

    held = alt.Chart(pd.DataFrame([{"start": start, "end": end}])).mark_rect(
        color=CANDLE, opacity=0.15
    ).encode(x="start:T", x2="end:T")

    def plan_line(price: float, color: str, label: str) -> List[alt.Chart]:
        data = pd.DataFrame([{"start": start, "end": end, "price": price, "label": label}])
        line = alt.Chart(data).mark_rule(color=color, strokeWidth=2).encode(
            x="start:T", x2="end:T", y="price:Q",
            tooltip=[alt.Tooltip("label:N", title=" "),
                     alt.Tooltip("price:Q", title="Price", format=",.4~f")],
        )
        text = alt.Chart(data).mark_text(
            align="right", baseline="middle", dx=-6, color=ink
        ).encode(x="start:T", y="price:Q", text="label:N")
        return [line, text]

    marks = pd.DataFrame([
        {"time": trade.entry_time, "price": trade.entry_price, "kind": "Buy", "label": "Buy"},
        {"time": trade.exit_time, "price": trade.exit_price, "kind": "Sell", "label": "Sell"},
    ])
    tooltip = [alt.Tooltip("kind:N", title=" "), alt.Tooltip("time:T", title="Date"),
               alt.Tooltip("price:Q", title="Price", format=",.4~f")]
    buy = alt.Chart(marks).transform_filter("datum.kind === 'Buy'").mark_point(
        shape="triangle-up", filled=True, size=160, opacity=1, color=ink, stroke=surface, strokeWidth=2
    ).encode(x="time:T", y="price:Q", tooltip=tooltip)
    sell = alt.Chart(marks).transform_filter("datum.kind === 'Sell'").mark_point(
        shape="triangle-down", filled=True, size=160, opacity=1, color=ink, stroke=surface, strokeWidth=2
    ).encode(x="time:T", y="price:Q", tooltip=tooltip)
    # Stop/Target left of the shaded period, Buy above its marker, Sell to the
    # right of its marker: stays readable when the trade's prices sit close.
    buy_label = alt.Chart(marks).transform_filter("datum.kind === 'Buy'").mark_text(
        align="center", baseline="bottom", dy=-12, color=ink, fontWeight="bold"
    ).encode(x="time:T", y="price:Q", text="label:N")
    sell_label = alt.Chart(marks).transform_filter("datum.kind === 'Sell'").mark_text(
        align="left", baseline="middle", dx=12, color=ink, fontWeight="bold"
    ).encode(x="time:T", y="price:Q", text="label:N")

    layers = [held] + sr_layers(window, levels, theme)
    layers += plan_line(trade.stop, RESISTANCE, "Stop")
    layers += plan_line(trade.target, SUPPORT, "Target")
    layers += [buy, sell, buy_label, sell_label]
    return alt.layer(*layers).properties(height=height)


def trade_summary(bars: List[Bar], trade: BounceTrade) -> str:
    """One line with the numbers behind a trade chart, incl. what happened after the sell."""
    move = trade.exit_price / trade.entry_price - 1
    days = f"{trade.days_held} day{'s' if trade.days_held != 1 else ''}"
    parts = [
        f"Bought {trade.entry_time:%Y-%m-%d} at {_fmt(trade.entry_price)}",
        f"sold {trade.exit_time:%Y-%m-%d} at {_fmt(trade.exit_price)} "
        f"({EXIT_LABEL[trade.exit_reason]}, {days})",
        f"price {move:+.1%}, account {trade.return_pct:+.2%} at {trade.position_size:.2f}x",
        f"stop {_fmt(trade.stop)}, target {_fmt(trade.target)}",
    ]
    index = {b.timestamp: i for i, b in enumerate(bars)}
    exit_i = index[trade.exit_time]
    after = [
        f"{n} days later {bars[exit_i + n].close / trade.exit_price - 1:+.1%}"
        for n in (5, 15) if exit_i + n < len(bars)
    ]
    if after:
        parts.append("after the sell: " + ", ".join(after))
    return " · ".join(parts)
