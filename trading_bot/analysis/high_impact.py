"""High-impact account classification.

When a post comes from a politician, regulator, central bank, executive, or
official company account, we apply special handling:

  * detect topics (tariffs, sanctions, rates, crypto regulation, etc.)
  * map topics to potentially affected sectors / tickers / crypto
  * distinguish official announcements from rumors / jokes / campaign rhetoric
  * require additional confirmation before trading unless the source is an
    official government or company account
  * flag politically sensitive or ambiguous posts as "watch only"

The output is consumed by the scoring model and the strategy layer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from ..config import get_settings


# -----------------------------------------------------------------------------
# Topic -> bias / impact / affected universe
# -----------------------------------------------------------------------------

@dataclass
class TopicImpact:
    """Describes the expected market impact of a high-impact-account topic."""

    topic: str
    bias: int                              # -1 bearish, 0 unclear, +1 bullish (broadly)
    impact: float                          # base magnitude 0..1
    affected_stocks: List[str] = field(default_factory=list)
    affected_crypto: List[str] = field(default_factory=list)
    sector_note: str = ""


TOPIC_RULES: Dict[str, Dict] = {
    "tariffs": {
        "patterns": [r"\btariff", r"\btrade war", r"\bimport tax", r"\bduties on\b"],
        "bias": -1, "impact": 0.75,
        "affected_stocks": ["AAPL", "TSLA", "NVDA", "AMZN", "XOM"],
        "affected_crypto": ["BTC", "ETH"],
        "sector_note": "importers, semis, autos, retailers, shipping, USD/gold reactive",
    },
    "sanctions": {
        "patterns": [r"\bsanction", r"\bembargo", r"\boil ban\b"],
        "bias": -1, "impact": 0.75,
        "affected_stocks": ["XOM", "EQNR.OL"],
        "affected_crypto": ["BTC"],
        "sector_note": "energy, shipping, defense; commodities reactive",
    },
    "crypto_regulation": {
        "patterns": [r"\bcrypto\b.*\bregulat", r"\bdigital asset\b.*\b(rule|law|bill)\b",
                     r"\bstablecoin\b.*\b(rule|law|bill|ban)\b", r"\bsec\b.*\bcrypto\b"],
        "bias": 0, "impact": 0.8,
        "affected_stocks": ["COIN", "MSTR"],
        "affected_crypto": ["BTC", "ETH", "SOL", "XRP"],
        "sector_note": "exchanges, miners, ETFs, stablecoin issuers",
    },
    "etf_approval_crypto": {
        "patterns": [r"\b(spot )?(btc|eth|ethereum|bitcoin) etf\b.*\bapprov",
                     r"\bcrypto etf\b.*\bapprov"],
        "bias": +1, "impact": 0.9,
        "affected_stocks": ["COIN", "MSTR"],
        "affected_crypto": ["BTC", "ETH"],
        "sector_note": "spot crypto ETF approval is unambiguously bullish for the underlying",
    },
    "interest_rates": {
        "patterns": [r"\bfed\b.*\brate", r"\binterest rate", r"\brate (cut|hike)",
                     r"\bdovish\b", r"\bhawkish\b", r"\bfomc\b"],
        "bias": 0, "impact": 0.7,
        "affected_stocks": ["JPM", "MSFT"],
        "affected_crypto": ["BTC", "ETH"],
        "sector_note": "rates move broad risk; banks, growth tech, crypto reactive",
    },
    "antitrust": {
        "patterns": [r"\bantitrust", r"\bmonopoly\b", r"\bbreak up\b.*\b(google|apple|amazon|meta)\b",
                     r"\bftc\b.*\b(sue|sues|suing|lawsuit)\b"],
        "bias": -1, "impact": 0.65,
        "affected_stocks": ["GOOGL", "AAPL", "AMZN", "META"],
        "affected_crypto": [],
        "sector_note": "big-tech mega caps",
    },
    "defense_spending": {
        "patterns": [r"\bdefen[sc]e (spending|budget)\b", r"\bmilitary aid\b",
                     r"\bweapons (deal|sale|contract)\b"],
        "bias": +1, "impact": 0.6,
        "affected_stocks": ["PLTR"],
        "affected_crypto": [],
        "sector_note": "defense primes, prime contractors",
    },
    "energy_oil": {
        "patterns": [r"\boil (sanction|embargo|price|ban)\b", r"\bopec\b",
                     r"\bgas pipeline\b", r"\bspr\b"],
        "bias": 0, "impact": 0.6,
        "affected_stocks": ["XOM", "EQNR.OL", "AKERBP.OL"],
        "affected_crypto": [],
        "sector_note": "integrated oil, E&P, refiners; inflation linkage",
    },
    "taxes": {
        "patterns": [r"\b(corporate )?tax (hike|cut|increase|reduction)\b",
                     r"\bbuyback tax\b", r"\bdividend tax\b"],
        "bias": 0, "impact": 0.55,
        "affected_stocks": [],
        "affected_crypto": [],
        "sector_note": "broad index reactive; sector-specific if specified",
    },
    "geopolitics": {
        "patterns": [r"\b(war|invasion|attack|missile|airstrike|coup|nuclear)\b",
                     r"\btaiwan strait\b", r"\bblockade\b"],
        "bias": -1, "impact": 0.8,
        "affected_stocks": [],
        "affected_crypto": ["BTC"],
        "sector_note": "risk-off broadly; defense up, EM down, oil/gold reactive",
    },
    "company_specific": {
        # Detected dynamically by entity extractor — placeholder so it appears in topic list.
        "patterns": [],
        "bias": 0, "impact": 0.5,
        "affected_stocks": [], "affected_crypto": [],
        "sector_note": "specific company mentioned by the high-impact account",
    },
}


# -----------------------------------------------------------------------------
# Tone classifiers (joke / threat / policy / official / campaign / rumor)
# -----------------------------------------------------------------------------

_JOKE_HINTS = [r"\b(lol|lmao|kidding|joking|sarcasm|/s)\b", r"😂|🤣|😜"]
_THREAT_HINTS = [r"\b(will|going to)\b.*\b(impose|introduce|sign|order)\b",
                 r"\b(massive|huge|biggest ever)\b.*\b(tariff|sanction)\b"]
_OFFICIAL_HINTS = [r"\bexecutive order\b", r"\bofficial statement\b",
                   r"\bissued\b", r"\bsigned (today|into law)\b",
                   r"\bannouncement\b", r"\bpress release\b"]
_CAMPAIGN_HINTS = [r"\b(rally|vote|election|campaign|make america|fight for)\b"]
_RUMOR_HINTS = [r"\b(rumor|allegedly|reportedly|sources say|unconfirmed|may|could|might)\b"]


def _matches_any(text: str, patterns: List[str]) -> bool:
    return any(re.search(p, text, re.I) for p in patterns)


# -----------------------------------------------------------------------------
# Public types
# -----------------------------------------------------------------------------

@dataclass
class HighImpactClassification:
    """Result of analyzing a post from a high-impact account."""

    is_high_impact: bool
    account_username: Optional[str] = None
    account_category: Optional[str] = None
    account_credibility: float = 0.0
    official_account: bool = False
    detected_topics: List[str] = field(default_factory=list)
    tone: str = "neutral"  # neutral | policy | threat | joke | campaign | rumor | official
    bias: int = 0
    expected_impact: float = 0.0
    affected_stocks: List[str] = field(default_factory=list)
    affected_crypto: List[str] = field(default_factory=list)
    reasoning: List[str] = field(default_factory=list)
    requires_extra_confirmation: bool = True
    watch_only: bool = False


def classify_high_impact_post(
    author: Optional[str],
    text: str,
) -> HighImpactClassification:
    """Classify a post; returns is_high_impact=False if author isn't on the list."""
    settings = get_settings()
    accounts = settings.high_impact_accounts

    if not author:
        return HighImpactClassification(is_high_impact=False)

    # Normalize "@user", "u/user", or plain username
    key = author.lstrip("@").split("/")[-1]
    meta = None
    for username, m in accounts.items():
        if username.lower() == key.lower():
            meta = m
            key = username
            break
    if meta is None:
        return HighImpactClassification(is_high_impact=False)

    text = text or ""

    # Topic detection
    topics: List[str] = []
    bias_sum = 0
    impact_max = 0.0
    affected_stocks: Set[str] = set()
    affected_crypto: Set[str] = set()
    reasoning: List[str] = [
        f"Post from {meta.get('name', key)} ({meta.get('category', 'other')})."
    ]
    for topic, rule in TOPIC_RULES.items():
        if not rule.get("patterns"):
            continue
        if any(re.search(p, text, re.I) for p in rule["patterns"]):
            topics.append(topic)
            bias_sum += rule["bias"]
            impact_max = max(impact_max, rule["impact"])
            affected_stocks.update(rule.get("affected_stocks", []))
            affected_crypto.update(rule.get("affected_crypto", []))
            reasoning.append(
                f"Topic '{topic}' matched: bias={rule['bias']:+d}, "
                f"impact={rule['impact']:.2f}. {rule['sector_note']}"
            )

    # Tone
    if _matches_any(text, _OFFICIAL_HINTS):
        tone = "official"
    elif _matches_any(text, _RUMOR_HINTS):
        tone = "rumor"
    elif _matches_any(text, _JOKE_HINTS):
        tone = "joke"
    elif _matches_any(text, _THREAT_HINTS):
        tone = "threat"
    elif _matches_any(text, _CAMPAIGN_HINTS):
        tone = "campaign"
    elif topics:
        tone = "policy"
    else:
        tone = "neutral"

    # Tone modifies impact
    tone_mods = {
        "official": 1.10,
        "policy":   1.00,
        "threat":   0.75,
        "campaign": 0.55,
        "rumor":    0.40,
        "joke":     0.10,
        "neutral":  0.50,
    }
    expected_impact = min(1.0, impact_max * tone_mods.get(tone, 0.5))

    # Official accounts: less extra confirmation needed.
    official = bool(meta.get("official", False))
    requires_extra_confirmation = not official or tone in {"joke", "campaign", "rumor"}
    watch_only = tone in {"joke", "campaign", "rumor"} or (
        not official and meta.get("category") in {"politician", "ceo", "crypto_exec"}
    )

    reasoning.append(f"Tone classified as '{tone}'.")
    if watch_only:
        reasoning.append(
            "Flagged as WATCH-ONLY: unofficial / ambiguous / political — "
            "requires strong corroboration before any trade."
        )
    if requires_extra_confirmation:
        reasoning.append(
            "Requires additional independent confirmation before trading."
        )

    return HighImpactClassification(
        is_high_impact=True,
        account_username=key,
        account_category=meta.get("category"),
        account_credibility=float(meta.get("credibility", 0.5)),
        official_account=official,
        detected_topics=topics,
        tone=tone,
        bias=(1 if bias_sum > 0 else -1 if bias_sum < 0 else 0),
        expected_impact=expected_impact,
        affected_stocks=sorted(affected_stocks),
        affected_crypto=sorted(affected_crypto),
        reasoning=reasoning,
        requires_extra_confirmation=requires_extra_confirmation,
        watch_only=watch_only,
    )
