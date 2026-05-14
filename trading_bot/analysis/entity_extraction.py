"""Lightweight entity extraction.

We do NOT depend on a heavyweight NLP stack for the MVP. Instead we use:
  * a regex pass to find $TICKER / TICKER patterns,
  * a known-name dictionary built from the configured watchlists,
  * a crypto-symbol dictionary.

This is intentionally simple. It is transparent, fast, and easy to extend.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Set

from ..config import get_settings

TICKER_RE = re.compile(r"(?<![A-Za-z0-9])\$?([A-Z]{1,5}(?:\.[A-Z]{1,3})?)(?![A-Za-z0-9])")

# Common-word filter so we don't classify "A", "I", "USA" as tickers.
TICKER_BLACKLIST: Set[str] = {
    "A", "I", "AN", "BE", "DO", "GO", "IS", "IT", "OK", "OR", "SO", "TO", "UP",
    "USA", "EU", "UK", "CEO", "CFO", "IPO", "ETF", "FDA", "SEC", "FED", "AI",
    "NEW", "FOR", "AT", "ON", "BY", "AS", "OF", "WE", "ME", "HE", "ALL", "THE",
    "WHO", "HOW", "GDP", "CPI", "PPI", "VAT",
}

# Asset-class hints for popular names not always inferable from ticker alone.
COMPANY_NAME_TO_TICKER: Dict[str, str] = {
    "apple":        "AAPL",
    "microsoft":    "MSFT",
    "tesla":        "TSLA",
    "nvidia":       "NVDA",
    "amazon":       "AMZN",
    "alphabet":     "GOOGL",
    "google":       "GOOGL",
    "meta":         "META",
    "facebook":     "META",
    "coinbase":     "COIN",
    "microstrategy":"MSTR",
    "palantir":     "PLTR",
    "jpmorgan":     "JPM",
    "exxon":        "XOM",
    "asml":         "ASML",
    "equinor":      "EQNR.OL",
    "dnb":          "DNB.OL",
    "telenor":      "TEL.OL",
    "mowi":         "MOWI.OL",
    "aker bp":      "AKERBP.OL",
    "yara":         "YAR.OL",
    "sap":          "SAP",
}

CRYPTO_KEYWORDS: Dict[str, str] = {
    "bitcoin": "BTC", "btc": "BTC", "$btc": "BTC",
    "ethereum": "ETH", "ether": "ETH", "eth": "ETH", "$eth": "ETH",
    "solana": "SOL", "sol": "SOL", "$sol": "SOL",
    "ripple": "XRP", "xrp": "XRP", "$xrp": "XRP",
    "dogecoin": "DOGE", "doge": "DOGE", "$doge": "DOGE",
    "cardano": "ADA", "ada": "ADA",
    "avalanche": "AVAX", "avax": "AVAX",
    "chainlink": "LINK", "link": "LINK",
}


@dataclass
class Entities:
    """Structured entity-extraction result."""

    tickers: List[str] = field(default_factory=list)
    companies: List[str] = field(default_factory=list)
    crypto: List[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.tickers or self.companies or self.crypto)


def extract_entities(text: str) -> Entities:
    """Extract tickers, company mentions and crypto symbols from free text."""
    if not text:
        return Entities()

    lower = text.lower()
    watch = set(get_settings().stock_watchlist)
    crypto_watch = set(get_settings().crypto_watchlist)

    tickers: Set[str] = set()
    for m in TICKER_RE.finditer(text):
        sym = m.group(1)
        if sym in TICKER_BLACKLIST:
            continue
        # Accept all-caps tokens of 2-5 chars, OR anything on the watchlist
        # (handles ".OL" suffixes), OR any token explicitly $-prefixed.
        prefixed = text[max(0, m.start() - 1): m.start()] == "$"
        if prefixed or sym in watch or (2 <= len(sym) <= 5 and sym.isupper()):
            tickers.add(sym)

    companies: Set[str] = set()
    for name, ticker in COMPANY_NAME_TO_TICKER.items():
        if name in lower:
            companies.add(name)
            tickers.add(ticker)

    crypto: Set[str] = set()
    for kw, sym in CRYPTO_KEYWORDS.items():
        # match as whole word
        if re.search(rf"\b{re.escape(kw)}\b", lower):
            crypto.add(sym)
    # also accept anything in the configured crypto watchlist if mentioned
    for sym in crypto_watch:
        if re.search(rf"\b{re.escape(sym)}\b", lower) or re.search(
            rf"\b{re.escape(sym.lower())}\b", lower
        ):
            crypto.add(sym)

    # Final pass: anything in stock watchlist that appears verbatim
    for sym in watch:
        if re.search(rf"(?<![A-Za-z0-9])\${re.escape(sym)}(?![A-Za-z0-9])", text) or re.search(
            rf"(?<![A-Za-z0-9]){re.escape(sym)}(?![A-Za-z0-9])", text
        ):
            tickers.add(sym)

    return Entities(
        tickers=sorted(tickers),
        companies=sorted(companies),
        crypto=sorted(crypto),
    )
