"""Transparent 0-100 opportunity scoring model.

Each factor produces a sub-score in [0, 1]. The factors are combined as a
weighted sum (weights come from config.scoring_weights) and multiplied by 100.
Every input that contributes to the score is also returned as a factor dict so
the strategy layer and dashboard can explain WHY a given score was produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from ..config import get_settings
from ..connectors.market_data import Quote
from .event_classifier import EventLabel
from .high_impact import HighImpactClassification


@dataclass
class ScoredOpportunity:
    """A scored trading opportunity for one asset."""

    symbol: str
    asset_class: str               # stock | crypto
    direction: str                 # buy | sell | watch | avoid
    total_score: float             # 0..100
    confidence: float              # 0..1
    risk_level: str                # low | medium | high
    factors: Dict[str, float] = field(default_factory=dict)
    reasoning: List[str] = field(default_factory=list)
    suggested_entry: Optional[float] = None
    suggested_stop_loss: Optional[float] = None
    suggested_take_profit: Optional[float] = None
    source_urls: List[str] = field(default_factory=list)
    contributing_event_ids: List[int] = field(default_factory=list)


@dataclass
class ScoringInputs:
    """Bundle of inputs the scorer needs for a (symbol, evidence) pair."""

    symbol: str
    asset_class: str
    sentiment: float                       # -1..1
    source_credibility: float              # 0..1
    event_timestamp: datetime
    event_label: Optional[EventLabel]
    high_impact: Optional[HighImpactClassification]
    quote: Optional[Quote]
    duplicate_confirmations: int           # 0..N (# independent corroborating sources)
    is_in_universe: bool
    is_blacklisted: bool
    is_rumor: bool


def _recency_score(ts: datetime, now: Optional[datetime] = None) -> float:
    """Score newer events higher. Linear decay over 24h, floor at 0.0."""
    now = now or datetime.utcnow()
    age = now - ts
    if age < timedelta(0):
        age = timedelta(0)
    if age >= timedelta(hours=24):
        return 0.0
    return max(0.0, 1.0 - age.total_seconds() / (24 * 3600))


def _norm(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def score_opportunity(inp: ScoringInputs) -> ScoredOpportunity:
    """Compute a transparent 0-100 score from the given evidence bundle."""
    settings = get_settings()
    weights = settings.scoring_weights

    reasoning: List[str] = []
    factors: Dict[str, float] = {}

    # --- Sentiment ---
    sentiment_sub = (inp.sentiment + 1.0) / 2.0  # map [-1,1] -> [0,1]
    factors["sentiment"] = sentiment_sub
    reasoning.append(f"Sentiment {inp.sentiment:+.2f} -> sub {sentiment_sub:.2f}")

    # --- Source credibility ---
    factors["source_credibility"] = max(0.0, min(1.0, inp.source_credibility))
    reasoning.append(f"Source credibility {inp.source_credibility:.2f}")

    # --- Recency ---
    rec = _recency_score(inp.event_timestamp)
    factors["recency"] = rec
    reasoning.append(f"Recency {rec:.2f}")

    # --- Relevance (event present + asset universe membership) ---
    relevance = 0.6 if inp.event_label is not None else 0.3
    if inp.is_in_universe:
        relevance += 0.4
    relevance = min(1.0, relevance)
    factors["relevance"] = relevance
    reasoning.append(f"Relevance {relevance:.2f}")

    # --- Expected impact (event + high-impact-account boost) ---
    expected = inp.event_label.expected_impact if inp.event_label else 0.0
    if inp.high_impact and inp.high_impact.is_high_impact:
        expected = max(expected, inp.high_impact.expected_impact)
    factors["expected_impact"] = max(0.0, min(1.0, expected))
    reasoning.append(f"Expected impact {expected:.2f}")

    # --- Volatility (we PREFER moderate vol; punish extreme vol) ---
    if inp.quote and inp.quote.daily_atr_pct is not None:
        atr = inp.quote.daily_atr_pct
        if atr <= 0.01:
            vol_sub = 0.4
        elif atr <= 0.05:
            vol_sub = 1.0
        elif atr <= settings.extreme_volatility_pct:
            vol_sub = 0.6
        else:
            vol_sub = 0.1
        reasoning.append(f"Daily ATR {atr:.2%} -> vol sub {vol_sub:.2f}")
    else:
        vol_sub = 0.5
    factors["volatility"] = vol_sub

    # --- Volume change ---
    if inp.quote:
        v = inp.quote.volume_change_pct
        # Reward elevated volume (confirmation), neutralize negative volume.
        vchg = _norm(v, 0.0, 2.0)
        factors["volume_change"] = vchg
        reasoning.append(f"Volume change {v:+.2%} -> sub {vchg:.2f}")
    else:
        factors["volume_change"] = 0.4

    # --- Price momentum (align with bias) ---
    if inp.quote:
        chg = inp.quote.daily_change_pct
        bias = 0
        if inp.event_label:
            bias = inp.event_label.bias
        if inp.high_impact and inp.high_impact.is_high_impact:
            bias = inp.high_impact.bias or bias
        aligned = chg * (1 if bias >= 0 else -1)
        pm = _norm(aligned, -0.02, 0.05)
        factors["price_momentum"] = pm
        reasoning.append(f"Price change {chg:+.2%} (bias {bias:+d}) -> sub {pm:.2f}")
    else:
        factors["price_momentum"] = 0.5

    # --- Duplicate confirmation ---
    dup_sub = _norm(inp.duplicate_confirmations, 0.0, 3.0)
    factors["duplicate_confirmation"] = dup_sub
    reasoning.append(f"Independent confirmations: {inp.duplicate_confirmations} -> sub {dup_sub:.2f}")

    # --- Risk level ---
    # Higher rumor flag / extreme vol / no confirmations / blacklist -> higher risk -> lower sub
    risk_penalty = 0.0
    if inp.is_rumor:
        risk_penalty += 0.4
    if not inp.is_in_universe:
        risk_penalty += 0.3
    if inp.is_blacklisted:
        risk_penalty += 1.0
    if inp.quote and inp.quote.daily_atr_pct and inp.quote.daily_atr_pct > settings.extreme_volatility_pct:
        risk_penalty += 0.3
    risk_sub = max(0.0, 1.0 - risk_penalty)
    factors["risk_level"] = risk_sub
    reasoning.append(f"Risk penalty {risk_penalty:.2f} -> risk sub {risk_sub:.2f}")

    # --- Weighted total ---
    total = 0.0
    for k, w in weights.items():
        total += factors.get(k, 0.0) * w
    total_score = round(total * 100, 2)

    # --- Direction ---
    bias = 0
    if inp.event_label:
        bias = inp.event_label.bias
    if inp.high_impact and inp.high_impact.is_high_impact:
        bias = inp.high_impact.bias or bias
    if inp.sentiment > 0.2:
        bias = bias if bias != 0 else +1
    elif inp.sentiment < -0.2:
        bias = bias if bias != 0 else -1

    direction = "watch"
    if inp.is_blacklisted or not inp.is_in_universe:
        direction = "avoid"
    elif total_score >= settings.min_score_to_trade and bias > 0:
        direction = "buy"
    elif total_score >= settings.min_score_to_trade and bias < 0:
        direction = "sell"  # short signals (we still won't auto-short in MVP)
    elif total_score >= settings.min_score_to_recommend:
        direction = "watch"
    else:
        direction = "avoid"

    # If a high-impact post is flagged watch-only and we have weak confirmation,
    # we override to watch regardless of score.
    if inp.high_impact and inp.high_impact.watch_only and inp.duplicate_confirmations < 1:
        direction = "watch"
        reasoning.append("Override: watch-only flag from high-impact classifier.")

    # Risk level label
    if risk_sub < 0.4 or (inp.quote and inp.quote.daily_atr_pct and inp.quote.daily_atr_pct > settings.extreme_volatility_pct):
        risk_label = "high"
    elif risk_sub < 0.7:
        risk_label = "medium"
    else:
        risk_label = "low"

    # Confidence (combines credibility, recency, duplicates, official boost)
    conf = (
        factors["source_credibility"] * 0.4
        + factors["duplicate_confirmation"] * 0.3
        + factors["recency"] * 0.2
        + (0.1 if (inp.high_impact and inp.high_impact.official_account) else 0.0)
    )
    if inp.is_rumor:
        conf *= 0.6
    confidence = round(max(0.0, min(1.0, conf)), 3)

    # Suggested levels
    entry = inp.quote.price if inp.quote else None
    stop = None
    take = None
    if entry:
        if bias >= 0:
            stop = round(entry * (1 - settings.default_stop_loss_pct), 4)
            take = round(entry * (1 + settings.default_take_profit_pct), 4)
        else:
            stop = round(entry * (1 + settings.default_stop_loss_pct), 4)
            take = round(entry * (1 - settings.default_take_profit_pct), 4)

    return ScoredOpportunity(
        symbol=inp.symbol,
        asset_class=inp.asset_class,
        direction=direction,
        total_score=total_score,
        confidence=confidence,
        risk_level=risk_label,
        factors=factors,
        reasoning=reasoning,
        suggested_entry=entry,
        suggested_stop_loss=stop,
        suggested_take_profit=take,
    )
