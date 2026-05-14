"""Tiny lexicon-based sentiment scorer.

Transparent and dependency-free. Returns a score in [-1.0, 1.0].
This is intentionally simple — replace with a real model later if needed.
"""

from __future__ import annotations

import re
from typing import Set


POSITIVE: Set[str] = {
    "beat", "beats", "surge", "surges", "surged", "jump", "jumps", "rally",
    "rallies", "soar", "soars", "soared", "growth", "record", "approve",
    "approved", "approval", "win", "wins", "won", "partnership", "deal",
    "acquisition", "acquire", "acquires", "buyback", "upgrade", "upgraded",
    "outperform", "raises", "raised", "strong", "expands", "expansion",
    "launch", "launched", "breakthrough", "milestone", "profitable",
    "profitability", "guidance raised", "all-time high", "ath", "bullish",
    "rate cut", "stimulus", "tax cut", "subsidy",
}

NEGATIVE: Set[str] = {
    "miss", "missed", "missing", "plunge", "plunges", "plunged", "drop", "drops",
    "fall", "falls", "fell", "slump", "slumps", "slumped", "lawsuit", "sue",
    "sued", "fraud", "investigation", "probe", "downgrade", "downgraded",
    "underperform", "loss", "losses", "bankrupt", "bankruptcy", "default",
    "hack", "hacked", "exploit", "breach", "delist", "delisted", "ban",
    "banned", "tariff", "tariffs", "sanction", "sanctions", "recall",
    "fine", "fined", "guilty", "settlement", "warning", "warns", "warned",
    "halt", "halted", "weak", "weakens", "weaker", "cut guidance", "bearish",
    "rate hike", "shutdown", "strike",
}

# Hedging words reduce magnitude — used as confidence dampener.
HEDGES: Set[str] = {
    "might", "may", "could", "possibly", "rumor", "rumored", "alleged",
    "reportedly", "unconfirmed", "speculation", "speculate", "considering",
    "weighing",
}

WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-]+")


def score_sentiment(text: str) -> float:
    """Return a sentiment score in [-1, 1]. 0.0 means neutral/unknown."""
    if not text:
        return 0.0
    words = [w.lower() for w in WORD_RE.findall(text)]
    if not words:
        return 0.0
    pos = sum(1 for w in words if w in POSITIVE)
    neg = sum(1 for w in words if w in NEGATIVE)
    hedge = sum(1 for w in words if w in HEDGES)
    if pos == 0 and neg == 0:
        return 0.0
    raw = (pos - neg) / max(1, pos + neg)
    if hedge:
        raw *= max(0.3, 1.0 - 0.15 * hedge)
    return max(-1.0, min(1.0, raw))


def is_rumor(text: str) -> bool:
    """Heuristic rumor flag — any hedge token marks the item as a rumor."""
    if not text:
        return False
    lower = text.lower()
    return any(h in lower for h in HEDGES)
