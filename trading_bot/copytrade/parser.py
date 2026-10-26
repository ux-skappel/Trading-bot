"""Turn a trader's free-text post into structured trade signals.

Built around headline-style posts, with classic signal formats as a fallback:

    GOOGLE LONG 🔼  Bruker 20X giring og SL på $327,3
    LONGER BLOOM OG WESTERN NÅ            (two trades)
    SOLGT HELE NÅ · SOLGT WESTERN · SL i profitt på $79900
    LONG $SOL  Entry 142-144  SL 138  TP1 150 TP2 158  10x  risk 1%
    Move SL to entry · Flytt stop til 145 · Close half SOL · Lukker alt BTC

A trade needs a coin/stock right next to LONG/SHORT (or a classic post with
entry/SL/TP labels), so chatter like "går long/short når vi nærmer oss" or a
promo code is never traded. Every decision is explained in ``Signal.notes``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

NUM = r"\d+(?:[.,]\d+)*"

KNOWN_COINS = {
    "BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "AVAX", "LINK", "BNB", "TON", "TRX", "DOT",
    "POL", "MATIC", "LTC", "BCH", "SHIB", "PEPE", "WIF", "BONK", "SUI", "APT", "ARB", "OP",
    "INJ", "TIA", "SEI", "NEAR", "ATOM", "FIL", "HBAR", "ICP", "RENDER", "RNDR", "FET", "TAO",
    "HYPE", "ENA", "ONDO", "JUP", "PYTH", "WLD", "STX", "IMX", "AAVE", "UNI", "MKR", "LDO",
    "CRV", "RUNE", "KAS", "FTM", "XLM", "ETC", "ORDI", "TRUMP", "FARTCOIN", "POPCAT", "MEW",
    "PENGU", "VIRTUAL", "KAITO", "BERA", "XMR", "ZEC", "ALGO", "GALA", "SAND", "MANA", "APE",
    "DYDX", "GMX", "PENDLE", "EIGEN", "ZRO", "STRK", "BLUR", "JTO", "BOME",
}
# Lower-case mentions only count for coins that aren't also ordinary words.
SAFE_LOWER = {
    "btc", "eth", "sol", "xrp", "doge", "ada", "avax", "bnb", "ltc", "pepe", "wif", "bonk",
    "sui", "arb", "inj", "tia", "sei", "atom", "aave", "matic", "shib", "trx", "kas", "ena",
    "ondo", "fet", "tao", "rndr", "hbar", "xlm", "fartcoin", "popcat", "pengu",
}
# Names -> the ticker Liquid/Hyperliquid uses (stocks and indices trade on the "xyz" market).
NAMES = {
    "BITCOIN": "BTC", "ETHEREUM": "ETH", "ETHER": "ETH", "SOLANA": "SOL", "RIPPLE": "XRP",
    "DOGECOIN": "DOGE", "CARDANO": "ADA", "AVALANCHE": "AVAX", "CHAINLINK": "LINK",
    "GOOGLE": "GOOGL", "ALPHABET": "GOOGL", "TESLA": "TSLA", "NVIDIA": "NVDA", "APPLE": "AAPL",
    "AMAZON": "AMZN", "MICROSOFT": "MSFT", "FACEBOOK": "META", "NETFLIX": "NFLX",
    "ROBINHOOD": "HOOD", "COINBASE": "COIN", "MICROSTRATEGY": "MSTR", "STRATEGY": "MSTR",
    "PALANTIR": "PLTR", "INTEL": "INTC", "SANDISK": "SNDK", "REDDIT": "RDDT", "BLOOM": "BE",
    "WESTERN": "WDC", "MICRON": "MU", "BROADCOM": "AVGO", "ORACLE": "ORCL", "SPACEX": "SPCX",
    "ANTHROPIC": "ANT", "COSTCO": "COST", "RIVIAN": "RIVN", "ALIBABA": "BABA",
    "GAMESTOP": "GME", "SAMSUNG": "SMSN", "HYNIX": "SKHX", "TSMC": "TSM", "CIRCLE": "CRCL",
    "GULL": "GOLD", "SØLV": "SILVER", "SOLV": "SILVER", "OLJE": "CL", "OIL": "CL",
    "BRENT": "BRENTOIL", "S&P500": "SP500", "SP500": "SP500", "SPX500": "SP500",
    "SPX": "SP500", "US500": "SP500", "NASDAQ": "XYZ100", "NAS100": "XYZ100",
}
MULTIWORD = [  # joined before matching so "WESTERN DIGITAL LONG" works
    (r"western\s+digital", "WDC"), (r"s\s*&\s*p\s*500", "S&P500"), (r"sk\s+hynix", "SKHX"),
    (r"bloom\s+energy", "BE"), (r"nasdaq\s*100", "XYZ100"),
]
# Words that are never a coin/stock even in capitals (English + Norwegian).
NOT_SYMBOLS = {
    "LONG", "SHORT", "TP", "TP1", "TP2", "TP3", "TP4", "SL", "ENTRY", "BUY", "SELL", "USD",
    "USDT", "USDC", "PERP", "SPOT", "LEV", "CROSS", "ISO", "ISOLATED", "STOP", "TARGET", "RISK",
    "BE", "CMP", "NOW", "DCA", "X", "ALL", "AND", "OG", "SIZE", "THE", "PNL", "ROI", "NFA",
    "DYOR", "ATH", "EP", "LIMIT", "MARKET", "CLOSE", "OPEN", "UPDATE", "NEW", "TRADE", "SIGNAL",
    "KJØP", "SELG", "MÅL", "NÅ", "I", "A", "TO", "AT", "ON", "IN", "IS", "IT", "MY", "ME",
    "OK", "FOMO", "HODL", "USA", "CPI", "FED", "FOMC", "NFP", "ETF", "SEC", "CEO", "LFG", "GM",
    "GN", "TA", "FA", "RR", "PM", "AM", "UTC", "CET", "EST", "NY", "EU", "UK", "NO", "YES",
    "VIP", "ALERT", "NEWS", "WAGMI", "NGMI", "IMO", "IMHO", "FYI", "BTW", "ASAP", "SWING",
    "SCALP", "HALF", "NOT", "DO", "DONT", "WAIT", "HOLD", "TPS", "SLS", "LEVELS", "ZONE",
    "HELE", "ALT", "RESTEN", "HALVPARTEN", "HALVE", "HALV", "HER", "FOR", "LETT", "ENKELT",
    "HØYERE", "LAVERE", "SOLGT", "KJØPT", "BOOM", "RISKY", "INNE", "MED", "PÅ", "SUPPORT",
    "RESISTANCE", "LONGER", "SHORTER", "GIRING", "ENNÅ", "INGEN", "MEGA", "BRA", "UT", "SER",
    "NY", "UKE", "OPP", "NED", "JEG", "VI", "DU", "ER", "DET", "SOM", "MEN", "AV", "TIL",
    "FRA", "NOE", "LITT", "MYE", "BARE", "OGSÅ", "GO", "LETS", "LET'S",
}


@dataclass
class Signal:
    """A parsed trade instruction (or ``info`` when there is none)."""

    action: str                          # open | close | move_sl | cancel | info
    symbol: Optional[str] = None
    side: Optional[str] = None           # long | short
    entry_low: Optional[float] = None
    entry_high: Optional[float] = None
    market: bool = False                 # enter at the current price
    stop_loss: Optional[float] = None
    take_profits: List[float] = field(default_factory=list)
    leverage: Optional[float] = None
    risk_pct: Optional[float] = None     # fraction of the account lost at the stop
    size_pct: Optional[float] = None     # fraction of the account used as margin
    size_scale: float = 1.0              # "in with 33% of the full position" -> 0.33
    close_fraction: float = 1.0          # for close: share of the position to close
    new_stop: Optional[float] = None     # for move_sl
    stop_to_entry: bool = False          # for move_sl: break-even
    notes: List[str] = field(default_factory=list)

    @property
    def tradable(self) -> bool:
        return self.action != "info"

    def summary(self) -> str:
        if self.action == "open":
            entry = ("market" if self.market or self.entry_low is None
                     else f"{self.entry_low:g}" if self.entry_low == self.entry_high
                     else f"{self.entry_low:g}-{self.entry_high:g}")
            parts = [f"OPEN {self.side.upper()} {self.symbol}", f"entry {entry}",
                     f"SL {self.stop_loss:g}" if self.stop_loss else "no SL"]
            if self.take_profits:
                parts.append("TP " + "/".join(f"{tp:g}" for tp in self.take_profits))
            if self.leverage:
                parts.append(f"{self.leverage:g}x")
            if self.risk_pct:
                parts.append(f"risk {self.risk_pct:.2%}")
            if self.size_pct:
                parts.append(f"size {self.size_pct:.2%}")
            if self.size_scale != 1.0:
                parts.append(f"{self.size_scale:.0%} of normal size")
            return " · ".join(parts)
        if self.action == "close":
            what = "all" if self.close_fraction >= 1 else f"{self.close_fraction:.0%}"
            return f"CLOSE {what} {self.symbol or '(replied/latest position)'}"
        if self.action == "move_sl":
            to = "entry (break-even)" if self.stop_to_entry else f"{self.new_stop:g}"
            return f"MOVE SL {self.symbol or '(replied/latest position)'} to {to}"
        if self.action == "cancel":
            return f"CANCEL {self.symbol or '(latest pending entry)'}"
        return "not a trade instruction"


def to_float(raw: str) -> float:
    """Parse 65,000 / 65 000 / 65.000,5 / 0,1234 / 1.5 into a float."""
    s = raw.strip().replace(" ", "").replace(" ", "")
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        head, *rest = s.split(",")
        if len(rest) > 1 or (len(rest[0]) == 3 and head not in ("", "0")):
            s = s.replace(",", "")
        else:
            s = s.replace(",", ".")
    elif s.count(".") > 1:
        s = s.replace(".", "")
    return float(s)


def _numbers(chunk: str) -> List[float]:
    """Numbers in a list like '150, 158 / 165 - 170 og 180' (comma+space separates)."""
    out = []
    for piece in re.split(r"\s*[/|&]\s*|\s+[-–]\s+|,\s+|\s+(?:and|og)\s+|\s+", chunk.strip()):
        piece = piece.strip().rstrip(".,;:!)")
        m = re.fullmatch(r"\$?(" + NUM + r")(k?)", piece, re.IGNORECASE)
        if m:
            value = to_float(m.group(1))
            out.append(value * 1000 if m.group(2) else value)
    return out


def _join_thousands(text: str) -> str:
    """'63 500' / '105 000' -> one number (Norwegian thousands separator).

    Only joins after a 1-2 digit group, or when the next group is '000', so a
    space-separated list like '150 158' stays two numbers.
    """
    text = re.sub(r"(?<![\w.,])(\d{1,2})[  ](\d{3})(?!\d|[.,]\d)", r"\1\2", text)
    return re.sub(r"(?<![\w.,])(\d{3})[  ](000)(?!\d|[.,]\d)", r"\1\2", text)


# "TP1", "tp 2", "target 3" — but in "tp 3.5k" the 3 is the price, not the index.
_IDX = r"(?:\s?\d(?![\d.,]\d))?(?!\d)"
TP_LABEL = re.compile(
    rf"(?:\btp{_IDX}|take[\s-]?profits?{_IDX}|\btargets?{_IDX}|\bmål{_IDX})\s*[:=@\-]?", re.IGNORECASE)
TP_SEGMENT_END = re.compile(
    r"\n|\b(?:sl|stop\w*|entry|lev\w*|risk|risiko|size|margin|tp|take[\s-]?profit|targets?|mål)\b"
    r"|\d+(?:[.,]\d+)?\s*x\b|\d+(?:[.,]\d+)?\s*%", re.IGNORECASE)


def _take_profits(text: str) -> List[float]:
    """Numbers after each TP/target label, up to the next label or line end."""
    values: List[float] = []
    for m in TP_LABEL.finditer(text):
        segment = text[m.end():m.end() + 120]
        end = TP_SEGMENT_END.search(segment)
        values += _numbers(segment[:end.start()] if end else segment)
    return values


def normalize_symbol(token: str) -> Optional[str]:
    """'GOOGLE' -> 'GOOGL', '$sol' -> 'SOL', 'HELE' -> None."""
    tagged = token.startswith(("$", "#"))
    t = token.strip().lstrip("$#").rstrip(".,!?:;)")
    up = t.upper()
    if up in NAMES:
        return NAMES[up]
    if not t or up in NOT_SYMBOLS:
        return None
    if not re.fullmatch(r"[A-Z0-9&]{2,10}", up) or up.isdigit():
        return None
    if tagged or up in KNOWN_COINS or t.isupper():
        return up
    return None


def _known_symbol(text: str) -> Optional[str]:
    """A $tagged ticker, known coin or known name — never a random capitalised word."""
    tagged = re.findall(r"[$#]([A-Za-z][A-Za-z0-9]{0,9})\b", text)
    if tagged:
        return normalize_symbol("$" + tagged[0])
    for token in re.findall(r"[A-Za-zÆØÅæøå&][A-Za-z0-9ÆØÅæøå&]{0,13}", text):
        up = token.upper()
        if up in NAMES and token[0].isupper():
            return NAMES[up]
        if (token.isupper() and up in KNOWN_COINS) or token in SAFE_LOWER:
            return up
    return None


def _symbol(text: str) -> Optional[str]:
    """Best single coin/stock mention anywhere (classic formats)."""
    tagged = re.findall(r"[$#]([A-Za-z][A-Za-z0-9]{0,9})\b", text)
    if tagged:
        return normalize_symbol("$" + tagged[0])
    pair = re.search(r"\b([A-Z0-9]{2,10})\s*(?:/|-)?\s*(?:USDT|USDC|USD|PERP)\b", text)
    if pair and pair.group(1) not in NOT_SYMBOLS:
        return pair.group(1)
    for token in re.findall(r"[A-Za-zÆØÅæøå&][A-Za-z0-9ÆØÅæøå&]{0,13}", text):
        up = token.upper()
        if up in NAMES and (token.isupper() or token[0].isupper()):
            return NAMES[up]
        if (token.isupper() and up in KNOWN_COINS) or token in SAFE_LOWER:
            return up
    for token in re.findall(r"\b[A-Z]{2,5}\b", text):
        if token not in NOT_SYMBOLS:
            return token
    return None


_SYMS = r"[$#]?[\w&.]+(?:\s+(?:og|and|&|\+)\s+[$#]?[\w&.]+)*"
HEADLINE_SYMBOL_FIRST = re.compile(rf"(?P<syms>{_SYMS})\s+(?P<side>long|short)\b", re.IGNORECASE)
HEADLINE_SIDE_FIRST = re.compile(
    rf"\b(?P<side>long(?:er)?|short(?:er)?|kjøper|buy(?:ing)?)\s+(?P<syms>{_SYMS})", re.IGNORECASE)


def _headline(text: str):
    """(side, [symbols]) for 'GOOGLE LONG' / 'LONGER BLOOM OG WESTERN' / 'LONG $SOL'."""
    for pattern in (HEADLINE_SYMBOL_FIRST, HEADLINE_SIDE_FIRST):
        for m in pattern.finditer(text):
            side = "short" if m.group("side").lower().startswith("short") else "long"
            tokens = re.split(r"\s+(?:og|and|&|\+)\s+", m.group("syms"), flags=re.IGNORECASE)
            if pattern is HEADLINE_SYMBOL_FIRST:
                tokens = [tokens[0].split()[-1]] + tokens[1:] if tokens else tokens
            symbols = [s for s in (normalize_symbol(t) for t in tokens) if s]
            if symbols and len(symbols) == len(tokens):
                return side, list(dict.fromkeys(symbols))
    return None, []


def _side(lower: str, has_levels: bool) -> Optional[str]:
    if re.search(r"long\s*/\s*short|short\s*/\s*long", lower):
        return None
    if re.search(r"\bshort\w*\b|📉|🔴", lower):
        return "short"
    if re.search(r"\blong\w*\b|\bbuy\w*\b|\bkjøp\w*|📈|🟢", lower):
        return "long"
    if has_levels and re.search(r"\bsell\b|\bselg\b", lower):
        return "short"
    return None


def _first(pattern: str, text: str) -> Optional[re.Match]:
    return re.search(pattern, text, re.IGNORECASE)


CLOSE_WORDS = re.compile(
    r"(?<!candle\s)(?<!candles\s)\b(?:close[ds]?|closing)\b"
    r"(?!\s+(?:to|til|by|near|above|below|over|under|green|red|nok|inn)\b)"
    r"|\b(?:solgt|selger|selg|sold|selling|exit(?:ed|ing)?|lukk\w*|stengt|cut(?:ting)?|kutt\w*|"
    r"tok\s+profitt|tar\s+profitt|tatt\s+profitt|ute\s+av|out\s+of|"
    r"take\s+(?:some\s+)?profits?\s+(?:now|here)|ta\s+(?:litt\s+)?gevinst|secur\w+\s+profits?)\b",
    re.IGNORECASE)
STOP_LEVEL = (r"(?:\bsl\b|stop[\s-]?loss|\bstop(?:p)?(?:[\s-]?loss)?\b|invalidation)\s*[:=@\-]?\s*"
              r"(?:at|på|ved|under|below|over|above|rundt)?\s*\$?(" + NUM + r")(k)?")


def _close_signal(raw: str, lower: str, close: re.Match) -> Signal:
    after = raw[close.end():close.end() + 80]
    symbol = None
    for token in re.findall(r"[$#]?[\w&.]+", after)[:6]:
        symbol = normalize_symbol(token)
        if symbol:
            break
    if symbol is None:
        symbol = _known_symbol(raw)
    sig = Signal("close", symbol=symbol)
    pct = (re.search(r"(?<![+\-\d])(\d{1,3})\s*%\s*(?:of|av)\b", lower)
           or re.search(r"\b(?:close|closing|lukk\w*|selger|solgt|sell\w*|take|ta)\b\D{0,15}?(?<![+\-\d])(\d{1,3})\s*%", lower))
    if re.search(r"\b(half|halv\w*|50\s*%)", lower):
        sig.close_fraction = 0.5
    elif pct and 0 < int(pct.group(1)) < 100:
        sig.close_fraction = int(pct.group(1)) / 100
    elif re.search(r"\b(partial\w*|delvis|some|litt)\b", lower):
        sig.close_fraction = 0.5
        sig.notes.append("partial close without a size: assumed half")
    sig.notes.append(f"close {sig.close_fraction:.0%}")
    return sig


def _open_details(sig: Signal, raw: str, lower: str) -> Signal:
    entry_m = _first(
        r"(?:\bentry(?:\s*zone)?\b|\bentries\b|\binngang\b|\bbuy\s*zone\b|\bep\b|\bkjøp\w*\s+(?:på|rundt|ved|i)\b|"
        r"\bbuy\w*\s+(?:at|@|around)\b|\blong\w*\s+(?:at|@|fra)\b|\bshort\w*\s+(?:at|@|fra)\b|@)"
        r"\s*[:=]?\s*\$?(" + NUM + r")(k)?(?:\s*(?:-|–|~|to|til)\s*\$?(" + NUM + r")(k)?)?", raw)
    if entry_m:
        lo = to_float(entry_m.group(1)) * (1000 if entry_m.group(2) else 1)
        hi = to_float(entry_m.group(3)) * (1000 if entry_m.group(4) else 1) if entry_m.group(3) else lo
        sig.entry_low, sig.entry_high = min(lo, hi), max(lo, hi)
    if not entry_m or re.search(r"\b(market|cmp|now|nå|markedspris|here|her)\b", lower):
        sig.market = True
        sig.notes.append("enter at the current price")
    stop_m = _first(STOP_LEVEL, raw)
    if stop_m:
        sig.stop_loss = to_float(stop_m.group(1)) * (1000 if stop_m.group(2) else 1)

    lev = (_first(r"\b(\d{1,3}(?:[.,]\d)?)\s*x\b", lower) or _first(r"\bx\s*(\d{1,3})\b", lower)
           or _first(r"\b(?:lev(?:erage)?|giring|gearing)\s*[:=]?\s*(\d{1,3})", lower))
    if lev:
        sig.leverage = to_float(lev.group(1))
    risk = (_first(r"\b(?:risk|risiko)\s*[:=]?\s*(" + NUM + r")\s*%", lower)
            or _first(r"(" + NUM + r")\s*%\s*(?:risk|risiko)", lower))
    if risk:
        sig.risk_pct = to_float(risk.group(1)) / 100
    size = (_first(r"\b(?:size|størrelse|margin)\s*[:=]?\s*(" + NUM + r")\s*%", lower)
            or _first(r"(" + NUM + r")\s*%\s*(?:of|av)\s*(?:the\s+|my\s+)?(?:portfolio|portefølje\w*|account|konto\w*|balance|saldo\w*)", lower))
    if size:
        sig.size_pct = to_float(size.group(1)) / 100
    partial = (_first(r"\binne\s+med\s+(\d{1,3})\s*%", lower)
               or _first(r"(\d{1,3})\s*%\s*(?:av|of)\s+(?:min\s+|my\s+)?(?:full\w*\s+)?(?:ønsket\s+|planned\s+|normal\s+|usual\s+)?posisjon\w*", lower)
               or _first(r"(\d{1,3})\s*%\s*(?:of\s+)?(?:my\s+)?(?:full\s+)?(?:planned\s+|normal\s+|usual\s+)?position", lower))
    if partial and 0 < int(partial.group(1)) < 100:
        sig.size_scale = int(partial.group(1)) / 100
        sig.notes.append(f"in with {sig.size_scale:.0%} of a full position")
    elif re.search(r"\b(ikke\s+stor|liten|lite)\s+posisjon|\bsmall\s+position", lower):
        sig.size_scale = 0.5
        sig.notes.append("small position: half the normal size")

    side, ref = sig.side, (sig.entry_high if sig.side == "long" else sig.entry_low)
    if sig.stop_loss and ref:
        if (side == "long" and sig.stop_loss >= sig.entry_low) or (side == "short" and sig.stop_loss <= sig.entry_high):
            sig.notes.append(f"stop {sig.stop_loss:g} is on the wrong side of the entry: ignored")
            sig.stop_loss = None
    good_tps = []
    for tp in _take_profits(raw):
        if ref and ((side == "long" and tp <= ref) or (side == "short" and tp >= ref)):
            sig.notes.append(f"target {tp:g} is on the wrong side of the entry: ignored")
        else:
            good_tps.append(tp)
    sig.take_profits = sorted(set(good_tps), reverse=(side == "short"))
    if sig.stop_loss is None:
        sig.notes.append("no stop loss in the post")
    return sig


def image_symbols(text: str, candidates) -> List[str]:
    """Which of ``candidates`` (e.g. open positions) an image's text names.

    Accepts tickers in any case and with exchange decorations (GOOGL-USD,
    xyz:GOOGL, BTC/USDC) as well as company names (Alphabet, SpaceX).
    """
    wanted = {c.upper() for c in candidates}
    found: List[str] = []
    for raw in re.findall(r"[A-Za-zÆØÅæøå&][\w&:\-/]{0,24}", text or ""):
        token = re.sub(r"(?i)^xyz:", "", raw)
        token = re.split(r"[-/]", token)[0]
        up = token.upper()
        symbol = NAMES.get(up, up)
        if symbol in wanted and symbol not in found:
            found.append(symbol)
    return found


def parse_signals(text: str) -> List[Signal]:
    """All instructions in one post (e.g. two trades, or a close plus a new trade)."""
    if not text or not text.strip():
        return [Signal("info", notes=["empty post"])]
    raw = _join_thousands(text)
    for pattern, replacement in MULTIWORD:
        raw = re.sub(pattern, replacement, raw, flags=re.IGNORECASE)
    lower = raw.lower()

    if re.search(r"\b(tp\s*\d|target\s*\d|mål\s*\d)\s*(hit|reached|nådd|tatt|done|✅)", lower):
        return [Signal("info", symbol=_symbol(raw), notes=["target-hit announcement; targets are tracked from prices"])]
    if re.search(r"\b(cancel\w*|avbryt\w*|kanseller\w*|ignore\s+(this|the)\s+(signal|trade|entry))\b", lower):
        return [Signal("cancel", symbol=_symbol(raw), notes=["cancel pending entry"])]

    head_side, head_symbols = _headline(raw)
    close = CLOSE_WORDS.search(raw)
    out: List[Signal] = []
    # A close word before the headline ("SOLGT TESLA, LONGER NVIDIA") is its own instruction;
    # one after it is part of the trade's text.
    if close and (not head_symbols or close.start() < _headline_start(raw)):
        out.append(_close_signal(raw, lower, close))

    if head_symbols:
        for sym in head_symbols:
            out.append(_open_details(Signal("open", symbol=sym, side=head_side), raw, lower))
        if len(head_symbols) > 1:
            for sig in out[-len(head_symbols):]:
                sig.notes.append(f"one of {len(head_symbols)} trades in the same post")
        return out
    if out:
        return out

    move = re.search(
        r"\b(?:move|moving|flytt\w*|set|sett|juster\w*|trail)\s+(?:the\s+|my\s+)?(?:sl|stop(?:p)?(?:[\s-]?loss)?)\b"
        r"|\b(?:sl|stop(?:p)?(?:[\s-]?loss)?)\s*(?:to|til|->|→|=|@)\s*(?:entry|be\b|break[\s-]?even|inngang|\$?" + NUM + ")",
        lower)
    stop_level = _first(STOP_LEVEL, raw) or _first(r"\b(?:sl|stop\w*)\b[^\d$\n]{0,30}?\$?(" + NUM + r")(k)?", raw)
    has_levels = bool(_first(r"\b(?:entry|inngang)\b", raw) or _take_profits(raw))
    if move or (stop_level and not has_levels and not _side(lower, False)):
        sig = Signal("move_sl", symbol=_symbol(raw) if re.search(r"[$#]\w|\b[A-Z]{2,5}\b", raw) else None)
        tail = lower[move.start():] if move else lower
        if re.search(r"\b(entry|be|break[\s-]?even|breakeven|inngang|kjøpspris)\b", tail):
            sig.stop_to_entry = True
            sig.notes.append("stop to entry (break-even)")
            return [sig]
        number = re.search(r"(?:to|til|->|→|=|@|på|at)\s*\$?(" + NUM + r")(k)?", tail) or stop_level
        if number:
            sig.new_stop = to_float(number.group(1)) * (1000 if number.group(2) else 1)
            sig.notes.append(f"new stop {sig.new_stop:g}")
            if sig.symbol and sig.symbol in NOT_SYMBOLS:
                sig.symbol = None
            return [sig]
        return [Signal("info", notes=["mentions moving the stop but no new level found"])]

    # Classic labelled signal (entry/SL/TP) without a headline.
    stop_m = _first(STOP_LEVEL, raw)
    if stop_m or has_levels:
        side, symbol = _side(lower, True), _symbol(raw)
        if side and symbol:
            return [_open_details(Signal("open", symbol=symbol, side=side), raw, lower)]
    side, symbol = _side(lower, False), _symbol(raw)
    missing = [n for n, v in (("direction (long/short) next to the coin", side and symbol), ) if not v]
    return [Signal("info", symbol=symbol, side=side, notes=["no trade: missing " + ", ".join(missing)])]


def _headline_start(text: str) -> int:
    starts = [m.start() for p in (HEADLINE_SYMBOL_FIRST, HEADLINE_SIDE_FIRST) for m in p.finditer(text)]
    return min(starts) if starts else len(text)


def parse_signal(text: str) -> Signal:
    """The first instruction in a post (see ``parse_signals`` for all of them)."""
    return parse_signals(text)[0]
