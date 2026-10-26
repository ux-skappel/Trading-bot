"""Automatic support and resistance detection.

The method is deliberately simple so every level can be explained:

1. Find confirmed swing highs and lows: a bar whose high (low) is the highest
   (lowest) within ``window`` bars on each side. The last ``window`` bars can
   never be pivots, so the detector never looks into the future.
2. Cluster pivot prices that lie within a tolerance band (half the average
   true range, clamped to 0.3%-3%) into one level. Highs and lows are clustered
   together because broken resistance often turns into support and vice versa.
3. Rank levels by number of touches and how recently they were touched.
4. Levels below the current price are support, levels above are resistance.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple


@dataclass
class Bar:
    """One OHLCV candle."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass
class PriceLevel:
    """A support or resistance level."""

    price: float
    kind: str              # support | resistance
    touches: int
    strength: float        # 0..1, touches + recency
    last_touch: datetime
    distance_pct: float    # signed distance from current price


@dataclass
class SupportResistance:
    """All levels detected for one price series."""

    symbol: str
    current_price: float
    tolerance_pct: float
    supports: List[PriceLevel] = field(default_factory=list)      # nearest first
    resistances: List[PriceLevel] = field(default_factory=list)   # nearest first

    @property
    def nearest_support(self) -> Optional[PriceLevel]:
        return self.supports[0] if self.supports else None

    @property
    def nearest_resistance(self) -> Optional[PriceLevel]:
        return self.resistances[0] if self.resistances else None

    def summary(self) -> str:
        """One-line human-readable description of the nearest levels."""
        parts = []
        s, r = self.nearest_support, self.nearest_resistance
        if s:
            parts.append(
                f"support {s.price:.4g} ({s.distance_pct:+.1%}, {s.touches} touch{'es' if s.touches != 1 else ''})"
            )
        else:
            parts.append("no support below (price under all known levels)")
        if r:
            parts.append(
                f"resistance {r.price:.4g} ({r.distance_pct:+.1%}, {r.touches} touch{'es' if r.touches != 1 else ''})"
            )
        else:
            parts.append("no resistance above (price discovery)")
        return f"{self.symbol} @ {self.current_price:.4g}: " + "; ".join(parts)


def average_true_range(bars: List[Bar], period: int = 14) -> float:
    """Average true range over the last ``period`` bars."""
    if not bars:
        return 0.0
    if len(bars) == 1:
        return bars[0].high - bars[0].low
    trs = [
        max(bar.high - bar.low, abs(bar.high - prev.close), abs(bar.low - prev.close))
        for prev, bar in zip(bars, bars[1:])
    ]
    recent = trs[-period:]
    return sum(recent) / len(recent)


def find_pivots(bars: List[Bar], window: int) -> List[Tuple[int, float]]:
    """Return (index, price) for every confirmed swing high and swing low."""
    pivots: List[Tuple[int, float]] = []
    for i in range(window, len(bars) - window):
        left = bars[i - window:i]
        right = bars[i + 1:i + window + 1]
        h, l = bars[i].high, bars[i].low
        # Strict on the left, inclusive on the right: a flat top/bottom yields
        # one pivot (its first bar) instead of several identical ones.
        if all(h > b.high for b in left) and all(h >= b.high for b in right):
            pivots.append((i, h))
        if all(l < b.low for b in left) and all(l <= b.low for b in right):
            pivots.append((i, l))
    return pivots


def _cluster(points: List[Tuple[int, float]], tol_pct: float) -> List[List[Tuple[int, float]]]:
    clusters: List[List[Tuple[int, float]]] = []
    for idx, price in sorted(points, key=lambda p: p[1]):
        if clusters:
            current = clusters[-1]
            mean = sum(p for _, p in current) / len(current)
            if mean > 0 and abs(price - mean) / mean <= tol_pct:
                current.append((idx, price))
                continue
        clusters.append([(idx, price)])
    return clusters


def detect_levels(
    bars: List[Bar],
    symbol: str = "",
    window: int = 5,
    max_levels: int = 3,
    tolerance_pct: Optional[float] = None,
) -> SupportResistance:
    """Detect support and resistance levels in a price series."""
    if not bars:
        return SupportResistance(symbol=symbol, current_price=0.0, tolerance_pct=0.0)

    current = bars[-1].close
    if tolerance_pct is None:
        atr = average_true_range(bars)
        tolerance_pct = min(0.03, max(0.003, 0.5 * atr / current)) if current > 0 else 0.01

    # Short series: shrink the window so we still find something.
    window = max(1, min(window, (len(bars) - 1) // 4))
    clusters = _cluster(find_pivots(bars, window), tolerance_pct)

    last_index = max(1, len(bars) - 1)
    supports: List[PriceLevel] = []
    resistances: List[PriceLevel] = []
    for c in clusters:
        price = sum(p for _, p in c) / len(c)
        last_idx = max(i for i, _ in c)
        touches = len(c)
        strength = 0.6 * min(touches, 4) / 4 + 0.4 * last_idx / last_index
        kind = "support" if price <= current else "resistance"
        level = PriceLevel(
            price=round(price, 6),
            kind=kind,
            touches=touches,
            strength=round(strength, 3),
            last_touch=bars[last_idx].timestamp,
            distance_pct=(price - current) / current if current else 0.0,
        )
        (supports if kind == "support" else resistances).append(level)

    def _top(levels: List[PriceLevel]) -> List[PriceLevel]:
        strongest = sorted(levels, key=lambda lv: lv.strength, reverse=True)[:max_levels]
        return sorted(strongest, key=lambda lv: abs(lv.distance_pct))

    return SupportResistance(
        symbol=symbol,
        current_price=current,
        tolerance_pct=tolerance_pct,
        supports=_top(supports),
        resistances=_top(resistances),
    )


# -----------------------------------------------------------------------------
# Stop loss / take profit from levels
# -----------------------------------------------------------------------------

@dataclass
class LevelTargets:
    """Stop loss and take profit derived from support/resistance."""

    stop_loss: float
    take_profit: float
    reward_risk: Optional[float]
    notes: List[str] = field(default_factory=list)
    stop_level: Optional[PriceLevel] = None     # None = percentage fallback
    target_level: Optional[PriceLevel] = None   # None = percentage fallback


def suggest_stop_take(
    sr: SupportResistance,
    entry: float,
    default_stop_pct: float,
    default_take_pct: float,
    max_stop_pct: float,
) -> LevelTargets:
    """Long-side stop just below support and target just below resistance.

    Falls back to the percentage defaults when no usable level exists.
    """
    notes: List[str] = []
    buffer = max(sr.tolerance_pct, 0.005)

    stop = entry * (1 - default_stop_pct)
    stop_level: Optional[PriceLevel] = None
    target_level: Optional[PriceLevel] = None
    usable = [s for s in sr.supports if s.price < entry and (entry - s.price * (1 - buffer)) / entry <= max_stop_pct]
    if usable:
        s = stop_level = usable[0]
        stop = s.price * (1 - buffer)
        notes.append(f"Stop {stop:.4g} just below support {s.price:.4g} ({s.touches} touches).")
    else:
        notes.append(f"No support within {max_stop_pct:.0%} — default {default_stop_pct:.0%} stop.")

    take = entry * (1 + default_take_pct)
    above = [r for r in sr.resistances if r.price > entry]
    if above:
        r = target_level = above[0]
        take = r.price * (1 - buffer)
        if take <= entry:
            take = r.price
        notes.append(f"Target {take:.4g} just below resistance {r.price:.4g} ({r.touches} touches).")
    else:
        notes.append(f"No resistance above — default {default_take_pct:.0%} target.")

    risk = entry - stop
    reward_risk = round((take - entry) / risk, 2) if risk > 0 else None
    if reward_risk is not None:
        notes.append(f"Reward/risk {reward_risk:.2f}.")
    return LevelTargets(
        stop_loss=round(stop, 4),
        take_profit=round(take, 4),
        reward_risk=reward_risk,
        notes=notes,
        stop_level=stop_level,
        target_level=target_level,
    )


# -----------------------------------------------------------------------------
# CSV input
# -----------------------------------------------------------------------------

_COLUMN_ALIASES: Dict[str, str] = {
    "timestamp": "timestamp", "date": "timestamp", "datetime": "timestamp",
    "time": "timestamp", "dato": "timestamp",
    "open": "open", "åpning": "open",
    "high": "high", "høy": "high", "høyeste": "high",
    "low": "low", "lav": "low", "laveste": "low",
    "close": "close", "adj close": "close", "price": "close", "last": "close",
    "siste": "close", "sluttkurs": "close", "kurs": "close",
    "volume": "volume", "volum": "volume",
}

_TS_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%d.%m.%Y", "%d.%m.%Y %H:%M", "%m/%d/%Y")


def _parse_ts(raw: str) -> datetime:
    raw = raw.strip()
    try:
        return datetime.fromisoformat(raw).replace(tzinfo=None)
    except ValueError:
        pass
    for fmt in _TS_FORMATS:
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    raise ValueError(f"unrecognised date: {raw!r}")


def bars_from_csv_text(text: str) -> List[Bar]:
    """Parse OHLCV CSV text into bars sorted oldest first.

    Accepts comma or semicolon separators (semicolon files may use decimal
    commas), common English/Norwegian column names, and close-only files.
    """
    first_line = text.lstrip().splitlines()[0] if text.strip() else ""
    delimiter = ";" if first_line.count(";") > first_line.count(",") else ","
    reader = csv.DictReader(io.StringIO(text.lstrip()), delimiter=delimiter)
    mapping = {
        name: _COLUMN_ALIASES[name.strip().lower()]
        for name in (reader.fieldnames or [])
        if name and name.strip().lower() in _COLUMN_ALIASES
    }
    if "timestamp" not in mapping.values() or "close" not in mapping.values():
        raise ValueError("CSV needs a date/timestamp column and a close/price column")

    def num(value: Optional[str]) -> Optional[float]:
        if value is None or not value.strip():
            return None
        v = value.strip().replace(" ", "")
        if delimiter == ";":
            v = v.replace(".", "").replace(",", ".") if "," in v else v
        return float(v)

    bars: List[Bar] = []
    for row in reader:
        fields = {mapping[k]: v for k, v in row.items() if k in mapping}
        close = num(fields.get("close"))
        if close is None or not fields.get("timestamp"):
            continue
        o = num(fields.get("open")) or close
        h = num(fields.get("high")) or max(o, close)
        l = num(fields.get("low")) or min(o, close)
        bars.append(Bar(
            timestamp=_parse_ts(fields["timestamp"]),
            open=o, high=h, low=l, close=close,
            volume=num(fields.get("volume")) or 0.0,
        ))
    bars.sort(key=lambda b: b.timestamp)
    return bars
