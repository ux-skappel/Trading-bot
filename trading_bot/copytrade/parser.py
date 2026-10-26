"""Turn a trader's free-text post into a structured trade signal.

Understands the common crypto-signal formats in English and Norwegian:

    LONG $SOL  Entry 142-144  SL 138  TP1 150 TP2 158  10x  risk 1%
    Kjøper BTC nå, stopp 63 500, mål 68 000 / 70 000
    Move SL to entry  ·  Flytt stop til 145  ·  Close half SOL  ·  Lukker alt BTC

Anything that isn't clearly a trade instruction comes back as ``info`` and is
never traded. Every decision is explained in ``Signal.notes``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

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
COIN_NAMES = {
    "bitcoin": "BTC", "ethereum": "ETH", "ether": "ETH", "solana": "SOL", "ripple": "XRP",
    "dogecoin": "DOGE", "cardano": "ADA", "avalanche": "AVAX", "chainlink": "LINK",
}
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
}
STOCKS_OK = re.compile(r"^[A-Z]{2,5}$")


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
            return " · ".join(parts)
        if self.action == "close":
            what = "all" if self.close_fraction >= 1 else f"{self.close_fraction:.0%}"
            return f"CLOSE {what} {self.symbol or '(latest position)'}"
        if self.action == "move_sl":
            to = "entry (break-even)" if self.stop_to_entry else f"{self.new_stop:g}"
            return f"MOVE SL {self.symbol or '(latest position)'} to {to}"
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


def _symbol(text: str) -> Optional[str]:
    tagged = re.findall(r"[$#]([A-Za-z][A-Za-z0-9]{0,9})\b", text)
    if tagged:
        return tagged[0].upper()
    pair = re.search(r"\b([A-Z0-9]{2,10})\s*(?:/|-)?\s*(?:USDT|USDC|USD|PERP)\b", text)
    if pair and pair.group(1) not in NOT_SYMBOLS:
        return pair.group(1)
    for token in re.findall(r"\b[A-Za-z][A-Za-z0-9]{0,9}\b", text):
        if (token.isupper() and token in KNOWN_COINS) or token in SAFE_LOWER:
            return token.upper()
    lower = text.lower()
    for name, sym in COIN_NAMES.items():
        if re.search(rf"\b{name}\b", lower):
            return sym
    for token in re.findall(r"\b[A-Z]{2,5}\b", text):
        if token not in NOT_SYMBOLS and STOCKS_OK.match(token):
            return token
    return None


def _side(lower: str, has_levels: bool) -> Optional[str]:
    if re.search(r"\bshort\w*\b|📉|🔴", lower):
        return "short"
    if re.search(r"\blong\w*\b|\bbuy\w*\b|\bkjøp\w*|📈|🟢", lower):
        return "long"
    if has_levels and re.search(r"\bsell\b|\bselg\b", lower):
        return "short"
    return None


def _first(pattern: str, text: str) -> Optional[re.Match]:
    return re.search(pattern, text, re.IGNORECASE)


def parse_signal(text: str) -> Signal:
    """Parse one post. Returns ``Signal(action='info')`` when it isn't an instruction."""
    if not text or not text.strip():
        return Signal("info", notes=["empty post"])
    raw = _join_thousands(text)
    lower = raw.lower()
    symbol = _symbol(raw)

    if re.search(r"\b(tp\s*\d|target\s*\d|mål\s*\d)\s*(hit|reached|nådd|tatt|done|✅)", lower):
        return Signal("info", symbol=symbol, notes=["target-hit announcement; targets are tracked from prices"])

    if re.search(r"\b(cancel\w*|avbryt\w*|kanseller\w*|ignore\s+(this|the)\s+(signal|trade|entry))\b", lower):
        return Signal("cancel", symbol=symbol, notes=["cancel pending entry"])

    move = re.search(
        r"\b(?:move|moving|flytt\w*|set|sett|juster\w*|trail)\s+(?:the\s+|my\s+)?(?:sl|stop(?:p)?(?:[\s-]?loss)?)\b"
        r"|\b(?:sl|stop(?:p)?(?:[\s-]?loss)?)\s*(?:to|til|->|→|=|@)\s*(?:entry|be\b|break[\s-]?even|inngang|\$?" + NUM + ")",
        lower,
    )
    if move:
        sig = Signal("move_sl", symbol=symbol)
        tail = lower[move.start():]
        if re.search(r"\b(entry|be|break[\s-]?even|breakeven|inngang|kjøpspris)\b", tail):
            sig.stop_to_entry = True
            sig.notes.append("stop to entry (break-even)")
            return sig
        number = re.search(r"(?:to|til|->|→|=|@)\s*\$?(" + NUM + r")", tail)
        if number:
            sig.new_stop = to_float(number.group(1))
            sig.notes.append(f"new stop {sig.new_stop:g}")
            return sig
        return Signal("info", symbol=symbol, notes=["mentions moving the stop but no new level found"])

    stop_m = _first(
        r"(?:\bsl\b|stop[\s-]?loss|\bstop(?:p)?(?:[\s-]?loss)?\b|invalidation)\s*[:=@\-]?\s*"
        r"(?:at|på|ved|under|below|over|above|rundt)?\s*\$?(" + NUM + r")(k)?", raw)
    tp_values = _take_profits(raw)
    entry_m = _first(
        r"(?:\bentry(?:\s*zone)?\b|\bentries\b|\binngang\b|\bbuy\s*zone\b|\bep\b|\bkjøp\w*\s+(?:på|rundt|ved|i)\b|"
        r"\bbuy\w*\s+(?:at|@|around)\b|\blong\w*\s+(?:at|@|fra|på)\b|\bshort\w*\s+(?:at|@|fra|på)\b|@)"
        r"\s*[:=]?\s*\$?(" + NUM + r")(k)?(?:\s*(?:-|–|~|to|til)\s*\$?(" + NUM + r")(k)?)?", raw)
    has_levels = bool(stop_m or entry_m or tp_values)
    side = _side(lower, has_levels)

    # Not "close to support" or "candle closed above": those are chart talk.
    close = re.search(
        r"(?<!candle\s)(?<!candles\s)\b(?:close[ds]?|closing)\b"
        r"(?!\s+(?:to|til|by|near|above|below|over|under|green|red|nok|inn)\b)"
        r"|\b(?:exit(?:ed|ing)?|lukk\w*|stengt|selger\s+(?:alt|resten|halv\w*)|sold|cut(?:ting)?|kutt\w*|"
        r"take\s+(?:some\s+)?profits?\s+(?:now|here)|ta\s+(?:litt\s+)?gevinst|secur\w+\s+profits?|out\s+of)\b",
        lower)
    if close and not (side and symbol and (stop_m or entry_m)):
        sig = Signal("close", symbol=symbol)
        # "30% of the position" / "close 30%", but not a "+12%" result.
        pct = (re.search(r"(?<![+\-\d])(\d{1,3})\s*%\s*(?:of|av)\b", lower)
               or re.search(r"\b(?:close|closing|lukk\w*|selger|sell\w*|take|ta)\b\D{0,15}?(?<![+\-\d])(\d{1,3})\s*%", lower))
        if re.search(r"\b(half|halv\w*|50\s*%)\b", lower):
            sig.close_fraction = 0.5
        elif pct and 0 < int(pct.group(1)) < 100:
            sig.close_fraction = int(pct.group(1)) / 100
        elif re.search(r"\b(partial\w*|delvis|some|litt)\b", lower):
            sig.close_fraction = 0.5
            sig.notes.append("partial close without a size: assumed half")
        sig.notes.append(f"close {sig.close_fraction:.0%}")
        return sig

    if not (side and symbol):
        missing = [n for n, v in (("direction (long/short)", side), ("coin", symbol)) if not v]
        return Signal("info", symbol=symbol, side=side, notes=["no trade: missing " + " and ".join(missing)])

    sig = Signal("open", symbol=symbol, side=side)
    if entry_m:
        k1 = 1000 if entry_m.group(2) else 1
        lo = to_float(entry_m.group(1)) * k1
        hi = to_float(entry_m.group(3)) * (1000 if entry_m.group(4) else 1) if entry_m.group(3) else lo
        sig.entry_low, sig.entry_high = min(lo, hi), max(lo, hi)
    if not entry_m or re.search(r"\b(market|cmp|now|nå|markedspris|here|her)\b", lower):
        sig.market = True
        sig.notes.append("enter at the current price")
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
    size = (_first(r"\b(?:size|størrelse|margin|posisjon\w*)\s*[:=]?\s*(" + NUM + r")\s*%", lower)
            or _first(r"(" + NUM + r")\s*%\s*(?:of|av)\s*(?:the\s+|my\s+)?(?:portfolio|portefølje\w*|account|konto\w*|balance|saldo\w*)", lower))
    if size:
        sig.size_pct = to_float(size.group(1)) / 100

    ref = sig.entry_high if side == "long" else sig.entry_low
    if sig.stop_loss and ref:
        if (side == "long" and sig.stop_loss >= sig.entry_low) or (side == "short" and sig.stop_loss <= sig.entry_high):
            sig.notes.append(f"stop {sig.stop_loss:g} is on the wrong side of the entry: ignored")
            sig.stop_loss = None
    good_tps = []
    for tp in tp_values:
        if ref and ((side == "long" and tp <= ref) or (side == "short" and tp >= ref)):
            sig.notes.append(f"target {tp:g} is on the wrong side of the entry: ignored")
        else:
            good_tps.append(tp)
    sig.take_profits = sorted(set(good_tps), reverse=(side == "short"))
    if sig.stop_loss is None:
        sig.notes.append("no stop loss in the post")
    return sig
