"""Classify news/event types by keyword patterns.

Each rule maps a regex to:
  * an event_type
  * an expected directional bias (+1 bullish, 0 unclear, -1 bearish)
  * an "expected_impact" magnitude in [0, 1]
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class EventLabel:
    """Classification output for a single normalized event."""

    event_type: str
    bias: int            # -1 bearish, 0 unclear, +1 bullish
    expected_impact: float  # 0..1


# (regex, event_type, bias, impact)
_RULES: List[Tuple[re.Pattern, str, int, float]] = [
    (re.compile(r"\bearnings\b|\bbeats?\b.*\bestimates?\b|\beps\b", re.I), "earnings", +1, 0.7),
    (re.compile(r"\bguidance (raised|lifted)\b|\braises (full[- ]year )?guidance\b", re.I), "earnings", +1, 0.7),
    (re.compile(r"\bguidance (cut|lowered)\b|\bprofit warning\b", re.I), "earnings", -1, 0.8),
    (re.compile(r"\b(lawsuit|sued|class action)\b", re.I), "lawsuit", -1, 0.5),
    (re.compile(r"\b(fda) (approval|approves|clearance|cleared)\b", re.I), "fda_approval", +1, 0.85),
    (re.compile(r"\b(fda) (rejects?|rejection)\b", re.I), "fda_rejection", -1, 0.85),
    (re.compile(r"\bcontract (award|win|signed)\b|\bdefense contract\b", re.I), "contract_win", +1, 0.6),
    (re.compile(r"\b(acquire|acquisition|takeover|merger|to buy)\b", re.I), "m&a", +1, 0.7),
    (re.compile(r"\bbankruptcy\b|\bchapter (7|11)\b|\bdefault\b", re.I), "bankruptcy", -1, 0.9),
    (re.compile(r"\b(hack|hacked|exploit|breach|stolen)\b", re.I), "hack", -1, 0.7),
    (re.compile(r"\b(etf) (approval|approves|approved|launched)\b", re.I), "etf_approval", +1, 0.8),
    (re.compile(r"\bregulation\b|\bregulatory\b|\bcrackdown\b", re.I), "regulation", 0, 0.6),
    (re.compile(r"\b(tariff|tariffs|sanction|sanctions)\b", re.I), "tariffs_sanctions", -1, 0.75),
    (re.compile(r"\b(partnership|partner with|alliance)\b", re.I), "partnership", +1, 0.4),
    (re.compile(r"\b(ceo) (steps down|resigns|fired|replaced)\b|\bnew ceo\b", re.I), "ceo_change", 0, 0.5),
    (re.compile(r"\b(production halt|recall|delay)\b", re.I), "production_issue", -1, 0.6),
    (re.compile(r"\b(war|invasion|attack|missile|airstrike|coup)\b", re.I), "geopolitical", -1, 0.7),
    (re.compile(r"\b(upgrade|outperform|overweight|buy rating)\b", re.I), "analyst_upgrade", +1, 0.35),
    (re.compile(r"\b(downgrade|underperform|underweight|sell rating)\b", re.I), "analyst_downgrade", -1, 0.4),
    (re.compile(r"\b(rate cut|cuts? rates|dovish)\b", re.I), "monetary_policy", +1, 0.55),
    (re.compile(r"\b(rate hike|raises? rates|hawkish)\b", re.I), "monetary_policy", -1, 0.55),
]


def classify_event(text: str) -> Optional[EventLabel]:
    """Return the first matching event classification or None."""
    if not text:
        return None
    for pat, event_type, bias, impact in _RULES:
        if pat.search(text):
            return EventLabel(event_type=event_type, bias=bias, expected_impact=impact)
    return None
