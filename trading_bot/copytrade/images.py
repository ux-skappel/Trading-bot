"""What the text read from one of his pictures says about a position.

Two kinds of picture are trusted:

* a **Liquid trade card** ("• Liquid | TSLA | PERP 10x LONG | CURRENT RETURN |
  +50.74% | ENTRY | $376,18 | MARK | $395,27") — coin, side, leverage and prices;
* a **price chart** (TradingView, or Liquid's own chart) — only the coin in its
  title, which comes before any watchlist in the read text.

Anything else (news screenshots, SizeProp or other platforms' cards, memes)
says nothing about his Liquid positions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from .parser import KNOWN_COINS, NAMES, NOT_SYMBOLS, to_float


@dataclass
class Card:
    symbol: str
    side: str                    # long | short
    leverage: Optional[float]
    entry: Optional[float]
    mark: Optional[float]

    def describe(self) -> str:
        lev = f"{self.leverage:g}x " if self.leverage else ""
        return f"{self.symbol} {lev}{self.side}"


# Names Liquid and chart sites print instead of the ticker.
IMAGE_NAMES = {
    **NAMES, "HYPERLIQUID": "HYPE", "ETHENA": "ENA", "METADAO": "META", "KPEPE": "PEPE",
    "KBONK": "BONK", "KSHIB": "SHIB", "BINANCE": "BNB", "META": "META", "GOLD": "GOLD",
    "SILVER": "SILVER", "NASDAQ 100": "XYZ100", "S&P 500": "SP500", "SPACE EXPLORATION": "SPCX",
    "ADVANCED MICRO": "AMD", "ROCKET LAB": "RKLB",
}
_CARD_SIDE = re.compile(r"(\d{1,3}(?:[.,]\d+)?)\s*[xхX]\s*(LONG|SHORT)\b")
_MONEY = re.compile(r"\$\s?(\d[\d  .,]*\d|\d)")
_NAME = re.compile(r"[\W_]*([A-Za-z][A-Za-z0-9&]*(?:\s(?:100|500))?)")
CHART_MARKERS = re.compile(r"TradingView|TetherUS|Perpetual|U\.S\. Dollar|At close|\bLIVE\b\s*\|\s*\d+m"
                           r"|[+-]\d[\d.,]*\s*\|?\s*\(\s*[+-]?\d[\d.,]*\s*%")
_SKIP = {"TV", "USD", "USDT", "USDC", "LIVE", "PRE", "POST", "AT", "CLOSE", "INC", "CORP", "LTD", "ETF"}


CHART_PHRASES = [(re.compile(p, re.IGNORECASE), s) for p, s in (
    (r"s\s*&\s*p\s*500", "SP500"), (r"nasdaq\s*100", "XYZ100"), (r"space\s+exploration", "SPCX"),
    (r"advanced\s+micro", "AMD"), (r"binance\s+coin", "BNB"), (r"rocket\s+lab", "RKLB"))]


def _symbol_for(name: str, strict: bool = True) -> Optional[str]:
    up = re.sub(r"\s+", " ", name.strip().upper())
    if up in IMAGE_NAMES:
        return IMAGE_NAMES[up]
    first = re.split(r"[ .]", up)[0]
    if first in IMAGE_NAMES:
        return IMAGE_NAMES[first]
    first = re.sub(r"(USDT|USDC|USD|PERP)$", "", first) or first
    if first in IMAGE_NAMES:
        return IMAGE_NAMES[first]
    if re.fullmatch(r"[A-Z][A-Z0-9]{1,5}", first) and first not in _SKIP and not (strict and first in NOT_SYMBOLS):
        return first
    return None


def _known(symbol: Optional[str]) -> bool:
    return bool(symbol) and (symbol in KNOWN_COINS or symbol in IMAGE_NAMES.values())


def _precise(raw: str) -> Optional[float]:
    """A card price, or None when it is rounded too much to use ("$0,09" for DOGE)."""
    digits = re.sub(r"\D", "", raw).lstrip("0")
    return to_float(raw) if len(digits) >= 4 else None


def cards(text: Optional[str]) -> List[Card]:
    """Every Liquid trade card in the picture (a screenshot can hold several)."""
    found: List[Card] = []
    parts = re.split(r"\bLiquid\b", text or "")
    for chunk in parts[1:]:
        chunk = chunk[:260]
        side = _CARD_SIDE.search(chunk)
        name = _NAME.match(chunk)
        if not side or not name or side.start() > 80:
            continue
        symbol = _symbol_for(name.group(1), strict=False)   # the card's name slot: "BE" is Bloom
        if symbol is None:
            continue
        prices = [_precise(m.group(1)) for m in _MONEY.finditer(chunk[side.end():])][:2]
        found.append(Card(symbol, side.group(2).lower(), to_float(side.group(1)),
                          prices[0] if prices else None, prices[1] if len(prices) > 1 else None))
    return found


def chart_symbol(text: Optional[str]) -> Optional[str]:
    """The coin in a price chart's title, or None when it isn't a chart."""
    if not text or not CHART_MARKERS.search(text) or cards(text):
        return None
    fields = text.split("|")[:8]
    for known_only in (True, False):   # a name or known coin first, then a bare ticker (IREN)
        for field in fields:
            phrase = next((s for p, s in CHART_PHRASES if p.search(field)), None)
            if phrase:
                return phrase
            words = [w for w in re.sub(r"[^\w&./ ]", " ", field).split() if len(w) > 1 and not w[0].isdigit()]
            for size in (2, 1):
                symbol = _symbol_for(" ".join(words[:size]).rstrip(".,")) if words else None
                if symbol and (_known(symbol) if known_only else words[0].isupper()):
                    return symbol
    return None
