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
    "META": "META", "GOLD": "GOLD", "SILVER": "SILVER",
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
    "HØYERE", "LAVERE", "SOLGT", "KJØPT", "BOOM", "RISKY", "INNE", "MED", "PÅ", "SUPPORT", "SOLGTE",
    "SELGER", "LIQUID", "PROP", "PROPPEN", "SIZEPROP", "DE", "DEN", "DERE", "VIL", "HA", "GG", "MVH",
    "DISSE", "DENNE", "JA", "NEI", "TAKK", "MEG", "NOK", "TROR", "SÅ", "OM", "YOLO", "ATH", "ZZZ",
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
    confidence: str = "normal"           # very_risky | risky | normal | strong (from his wording)
    add: bool = False                    # "longer mer": adds to his position (symbol may come from a reply)
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
            parts = [f"{'ADD TO' if self.add else 'OPEN'} {self.side.upper()} {self.symbol or '(replied position)'}",
                     f"entry {entry}",
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
            if self.confidence != "normal":
                parts.append(CONFIDENCE_LABELS[self.confidence])
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
    if not re.fullmatch(r"[A-Z][A-Z0-9&]{1,9}", up):   # never "10X" or "4K"
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


SIDE_WORD = re.compile(r"\b(long(?:er|et|e)?|short(?:er|et|e)?|kjøper|buy(?:ing)?)\b", re.IGNORECASE)
_ADD = re.compile(r"[ \t]+(?:(?:en\s+god\s+del|mye|litt|enda)\s+)?(?:mer|more)\b", re.IGNORECASE)
_NEXT_TOKEN = re.compile(r"[ \t]*([$#]?[\w&.]+)")
_JOIN_AFTER = re.compile(r"[ \t]+(?:og|and|&|\+)[ \t]+", re.IGNORECASE)
_LAST_TOKEN = re.compile(r"([$#]?[\w&.]+)[ \t]+$")
_JOIN_BEFORE = re.compile(r"(?:^|[ \t])(?:og|and|&|\+)[ \t]+$", re.IGNORECASE)
CALL_VERB = re.compile(r"\b(?:velger|går|gå|kjører|tar|åpner|åpnet)\b", re.IGNORECASE)
HEDGE = re.compile(r"\b(?:tenker|vurderer|kanskje|tror|kan|skal|om|hvis|planlegger|fristende|kommer\s+til\s+å)\b",
                   re.IGNORECASE)


@dataclass
class _Head:
    """One 'SYMBOL(S) + LONG/SHORT' in a post."""

    side: str
    symbols: List[str]
    start: int
    end: int
    symbol_first: bool = False   # "GOOGLE LONG" rather than "LONG GOOGLE"
    add: bool = False            # "longer mer TESLA" / "longet mer"
    verb: bool = False           # "velger long META", "går long" on the same line


def _symbols_after(text: str, pos: int):
    symbols, end = [], pos
    m = _NEXT_TOKEN.match(text, pos)
    while m:
        symbol = normalize_symbol(m.group(1))
        if not symbol:
            break
        symbols.append(symbol)
        end = m.end()
        join = _JOIN_AFTER.match(text, end)
        m = _NEXT_TOKEN.match(text, join.end()) if join else None
    return symbols, end


def _symbols_before(text: str, pos: int):
    symbols, start = [], pos
    while True:
        m = _LAST_TOKEN.search(text[:start])
        symbol = normalize_symbol(m.group(1)) if m else None
        if not symbol:
            break
        symbols.insert(0, symbol)
        start = m.start()
        join = _JOIN_BEFORE.search(text[:start])
        if not join:
            break
        start = join.start()
    return symbols, start


def _headlines(text: str) -> List[_Head]:
    """Every 'GOOGLE LONG' / 'LONGER BLOOM OG WESTERN' / 'velger long A og B + short C'."""
    heads: List[_Head] = []
    for m in SIDE_WORD.finditer(text):
        if re.match(r"\s*/\s*(?:long|short)", text[m.end():], re.IGNORECASE) or \
                re.search(r"(?:long|short)\s*/\s*$", text[:m.start()], re.IGNORECASE):
            continue   # "går long/short når vi nærmer oss"
        side = "short" if m.group(1).lower().startswith("short") else "long"
        add = _ADD.match(text, m.end())
        before, b_start = _symbols_before(text, m.start())
        after, a_end = _symbols_after(text, add.end() if add else m.end())
        if before and not add:
            head = _Head(side, before, b_start, m.end(), symbol_first=True)
        elif after:
            head = _Head(side, after, m.start(), a_end, add=bool(add))
        elif add:
            head = _Head(side, [], m.start(), add.end(), add=True)
        else:
            continue
        line_start = text.rfind("\n", 0, head.start) + 1
        before_head = text[line_start:head.start]
        head.verb = bool(CALL_VERB.search(before_head)) and not HEDGE.search(before_head)
        heads.append(head)
    return heads


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


# "close" alone is mostly about candles ("1h close", "nydelig close"): a sell only with an object.
CLOSE_WORDS = re.compile(
    r"\b(?:close[ds]?|closing)\b(?=\s+(?:all|half|everything|here|now|the\s+(?:trade|position)|my|our|position|trade|"
    r"it\b|\d{1,3}\s*%|(?-i:[$#]?[A-Z]{2,10}\b)))"
    r"|\b(?:solgt\w*|selger|selges?|selg|sold|selling|exit(?:ed|ing)?(?!\s+plan)|lukk\w*|stengt|"
    r"cut(?:ting)?\s+(?:loss\w*|the\s+(?:trade|position)|it)|kutt(?:er|et)?\s+(?:tap\w*|posisjon\w*|denne|den|hele|alt)|"
    r"tok\s+profitt|tar\s+profitt|tatt\s+profitt|ute\s+av\s+(?:denne|den|traden|posisjonen|hele)|"
    r"out\s+of\s+(?:the|this|my)\s+(?:trade|position)|takk\s+for\s+meg|"
    r"take\s+(?:some\s+)?profits?\s+(?:now|here)|ta\s+(?:litt\s+)?gevinst|secur\w+\s+profits?)\b",
    re.IGNORECASE)
# "om vi lukker under X så selger jeg", "selger når vi treffer $X", "selger ved $X",
# "tror jeg selger ikveld", "kan selge 50%": a plan, not a sell now.
IF_BEFORE = re.compile(r"\b(?:om|hvis|dersom|if|unless|evt|eventuelt|kanskje|tror|vurderer|kan|skal|vel|"
                       r"kommer\s+til\s+å|planlegger)\b", re.IGNORECASE)
IF_AFTER = re.compile(r"[^.!?\n…]{0,25}?\b(?:når|om|hvis|dersom|if|when|once)\s+"
                      r"(?:vi|den|prisen|kursen|it|price|we|\$?\d)", re.IGNORECASE)
LATER = re.compile(r"[ \t]+(?:\w+[ \t]+){0,2}?(?:nok|evt|kanskje|i\s+dag\s+før|ikveld|i\s+kveld|i\s+morgen|"
                   r"før\s+(?:kl|jeg|leggetid)|etter\s+(?:noen|kl)|(?:ved|på|at|@)\s*\$?\d)", re.IGNORECASE)
PAST_CLOSE = re.compile(r"solgt\w*|sold|stengt|lukket|exited|closed", re.IGNORECASE)
# "for øyeblikket er jeg long X" / "fortsatt long" / "er long X": already in it.
STATUS = re.compile(r"\b(?:for\s+øyeblikket|per\s+nå|fortsatt|fremdeles|still|currently|sitter|er)\s+"
                    r"(?:\w+\s+){0,3}?(?:long|short)\b")
# "bruker 10X giring kl HH:MM", "går inn kl HH:MM", "går short nøyaktig kl HH": later.
FUTURE_ENTRY = re.compile(
    r"\b(?:går\s+inn|g[åa]r?\s+(?:long|short)|longer|shorter|kjøper|bruker\s+\d{1,3}\s*x(?:\s+giring)?)"
    r"[^.\n(]{0,25}?\b(?:kl\.?\s*\d{1,2}[:.]\d{2}|i\s+morgen)\b|\bnøyaktig\s+kl\b"
    r"|\b(?:kl\.?\s*\d{1,2}[:.]\d{2}|i\s+morgen)\b[^.\n(]{0,25}?\b(?:går\s+inn|longer|shorter)\b", re.IGNORECASE)
# SizeProp/prop-firm accounts are not Liquid: those clauses are left out.
OTHER_PLATFORM = re.compile(r"\bsize\s*prop\w*|\bprop(?:pen|firm\w*)?\b|\bpå\s+size\b|\bstonkex\b|\bpå\s+fomo\b",
                            re.IGNORECASE)


def _conditional(raw: str, close: re.Match) -> bool:
    start = max(raw.rfind(c, 0, close.start()) for c in ".!?:;\n…") + 1
    if IF_BEFORE.search(raw[start:close.start()]) or IF_AFTER.match(raw, close.end()):
        return True
    return not PAST_CLOSE.fullmatch(close.group(0)) and bool(LATER.match(raw, close.end()))


def _liquid_only(text: str):
    """The post without clauses about SizeProp/prop accounts, and whether any were dropped."""
    lines, dropped = [], False
    for line in text.split("\n"):
        parts = re.split(r"(\s+\+\s+|,\s+)", line)
        kept = []
        for i in range(0, len(parts), 2):
            if OTHER_PLATFORM.search(parts[i]):
                dropped = True
            else:
                kept.append((parts[i - 1] if i and kept else "") + parts[i])
        lines.append("".join(kept))
    return "\n".join(lines), dropped


# A level, never leverage ("100x") or a percentage ("-10%").
_NO_X_PCT = r"(?![\d.,]*\s*(?:x\b|%))"
STOP_LEVEL = (r"(?:\bsl\b|stop[\s-]?loss|\bstop(?:p)?(?:[\s-]?loss)?\b|invalidation)[ \t]*[:=@\-]?[ \t]*"
              r"(?:at|på|ved|under|below|over|above|rundt)?[ \t]*\$?(" + NUM + r")(k)?" + _NO_X_PCT)
LOOSE_STOP = r"\b(?:sl|stop\w*)\b[^\d$\n-]{0,30}?\$?(" + NUM + r")(k)?" + _NO_X_PCT
MOVE_VERB = re.compile(
    r"\b(?:move|moving|flytt\w*|set|sett|juster\w*|trail|plasser\w*)\s+(?:the\s+|my\s+)?(?:sl|stop(?:p)?(?:[\s-]?loss)?)\b"
    r"|\b(?:sl|stop(?:p)?(?:[\s-]?loss)?)\s+(?:plassert|flyttet|satt)\b"
    r"|\b(?:sl|stop(?:p)?(?:[\s-]?loss)?)\s*(?:to|til|->|→|=|@)\s*(?:entry|be\b|break[\s-]?even|inngang|\$?" + NUM + ")",
    re.IGNORECASE)
# "COIN SL i profitt ved $X" on its own line (one per coin).
LINE_MOVE = re.compile(r"^[ \t]*([$#]?[A-Za-z][\w&]*)[ \t]+(?:sl|stop\w*)\b[^\n]*?\$?(" + NUM + r")(k)?" + _NO_X_PCT,
                       re.IGNORECASE | re.MULTILINE)

# How sure he sounds -> how big the copy is (multipliers in CopyConfig / .env).
CONFIDENCE_LABELS = {"very_risky": "very risky: much smaller size", "risky": "risky: smaller size",
                     "strong": "strong: bigger size"}
NEGATED = re.compile(r"\b(?:ikke|not|lite|lav)\s+(?:så\s+|so\s+|helt\s+|veldig\s+|very\s+)?"
                     r"(?:risky|risikab\w*|klink\w*|trygg\w*|bra|sterk\w*|perfekt\w*)")
VERY_RISKY = re.compile(
    r"\b(?:veldig|svært|super|ekstremt|ekstra|mega|sykt|very|extremely|highly)\s+risky\b"
    r"|\brisky\W*(?:⚠️?\W*){2,}|\bgambl\w*|\blotto\w*|\byolo\b")
RISKY = re.compile(
    r"\brisky\b|\brisikab\w*|\bhøy\s+risiko\b|\bhigh\s+risk\b|⚠|\bforsiktig\w*|\bcareful\b"
    r"|\b(?:ikke\s+(?:så\s+)?stor|liten|lite)\s+posisjon|\bsmall\s+position")
STRONG = re.compile(
    r"\bklink\w*|\b(?:mega|veldig|kjempe|super|sykt|helt)\s+(?:bra|sterk\w*|god\w*)\b"
    r"|\bser\s+(?:(?!ikke\b)\w+\s+)?(?:bra|sterk\w*|god|perfekt\w*)\s+ut\b|\bsterk\w*\s+support"
    r"|\b(?:liker?|love|god|good|great|fin)\s+(?:\w+\s+)?risk[\s/-]?reward\w*"
    r"|\bperfekt\w*|\btrygg\w*|\bhigh\s+conviction\b"
    r"|oversolgt\W+(?:\w+\W+){0,3}support|support\W+(?:\w+\W+){0,3}oversolgt")


# A real call has leverage, a stop, levels or 🔼/🔽 right after "HOOD LONG"; chat that
# mentions a side ("rapport for X … short") or a meme caption never does.
JOKE = re.compile(r"[😂🤣😆😅💀🙈]|\b(?:haha\w*|hehe\w*|lol|lmao|meme\w*)\b")
TRADE_MARK = re.compile(r"[🔼🔽📈📉]")


def _has_trade_details(sig: Signal, raw: str, headline_end: int) -> bool:
    return bool(sig.leverage or sig.stop_loss or sig.take_profits or sig.risk_pct or sig.size_pct
                or not sig.market or TRADE_MARK.search(raw, headline_end, headline_end + 12))


def _confidence(sig: Signal, lower: str) -> None:
    """Risky words make the copy smaller, solid ones ("klink support") bigger. Risky wins."""
    text = NEGATED.sub(" ", lower)
    strong = STRONG.search(text)
    for label, pattern in (("very_risky", VERY_RISKY), ("risky", RISKY)):
        m = pattern.search(text)
        if m:
            sig.confidence = label
            sig.notes.append(f"he says '{m.group(0).strip()}': {CONFIDENCE_LABELS[label]}"
                             + (f" (also '{strong.group(0)}', but risky wins)" if strong else ""))
            return
    if strong:
        sig.confidence = "strong"
        sig.notes.append(f"he says '{strong.group(0)}': {CONFIDENCE_LABELS['strong']}")


def _close_signal(raw: str, lower: str, close: re.Match, sold: Optional[str] = None) -> Signal:
    line_end = raw.find("\n", close.end())
    after = raw[close.end():line_end if line_end >= 0 else len(raw)][:80]   # the coin is on the same line
    symbol = sold
    for token in ([] if symbol else re.findall(r"[$#]?[\w&.]+", after)[:6]):
        symbol = normalize_symbol(token)
        if symbol:
            break
    if symbol is None:   # "fin long på X 🔼 selger hele nå"
        named = re.search(r"\b(?:long|short)\w*\s+(?:på|on|i|in)\s+([$#]?[\w&.]+)", raw, re.IGNORECASE)
        symbol = normalize_symbol(named.group(1)) if named else None
    sig = Signal("close", symbol=symbol)
    lev = re.search(r"\b(\d{1,3})\s*x\b", after, re.IGNORECASE)   # "solgt 100x, beholder 40x"
    if lev:
        sig.leverage = float(lev.group(1))
        sig.notes.append(f"his {lev.group(1)}x position")
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
    _confidence(sig, lower)

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


TICKER_IN_IMAGE = re.compile(
    r"\b([A-Z][A-Z0-9&]{1,9})(?=[-/]USD[CT]?\b|\s*\|?\s*(?:PERP|LONG|SHORT|Long|Short)\b|\s+\d{1,3}x\b)")


def image_tickers(text: str) -> List[str]:
    """Every ticker a trade card names (AMD-USD, xyz:AMD, 'AMD PERP 10x LONG'), held or not."""
    found: List[str] = []
    for token in TICKER_IN_IMAGE.findall(re.sub(r"(?i)\bxyz:", "", text or "")):
        symbol = NAMES.get(token, token)
        if symbol not in NOT_SYMBOLS and symbol not in found:
            found.append(symbol)
    return found


def _all_symbols(text: str) -> List[str]:
    """Every coin/stock the post names for sure ($TAG, known coin, company name, a lone first line)."""
    found = [normalize_symbol("$" + t) for t in re.findall(r"[$#]([A-Za-z][A-Za-z0-9]{0,9})\b", text)]
    for token in re.findall(r"[A-Za-zÆØÅæøå&][A-Za-z0-9ÆØÅæøå&]{0,13}", text):
        up = token.upper()
        if up in NAMES and token[0].isupper():
            found.append(NAMES[up])
        elif (token.isupper() and up in KNOWN_COINS) or token in SAFE_LOWER:
            found.append(up)
    first = text.strip().split("\n")[0].strip()
    if re.fullmatch(r"[$#]?[A-Z][A-Z0-9&]{1,9}", first):   # "NETFLIX" alone on the first line
        found.append(normalize_symbol(first))
    return [s for s in dict.fromkeys(found) if s]


PROSE_SIDE = re.compile(r"\b(?:gå|går|kjører|tar|velger)\s+(?:å\s+)?(?:gå\s+)?(long|short)\b", re.IGNORECASE)
LEVERAGE = re.compile(r"\b\d{1,3}(?:[.,]\d)?\s*x\b", re.IGNORECASE)


def _prose_call(raw: str, lower: str) -> Optional[Signal]:
    """'Rapport for <company> … velger å gå long med 10X giring' / '<COMPANY> … opp 🔼 … 10X giring'."""
    if not LEVERAGE.search(raw):
        return None
    m = PROSE_SIDE.search(raw)
    ups, downs = sum(raw.count(c) for c in "🔼📈"), sum(raw.count(c) for c in "🔽📉")
    if m:
        side = m.group(1).lower()
    elif ups and not downs and re.search(r"\b(?:opp|up|pump\w*)\b", lower):
        side = "long"
    elif downs and not ups and re.search(r"\b(?:ned|down|dump\w*)\b", lower):
        side = "short"
    else:
        return None
    symbols = _all_symbols(raw)
    if len(symbols) != 1:
        return None
    sig = _open_details(Signal("open", symbol=symbols[0], side=side), raw, lower)
    sig.notes.insert(0, f"call written out in words: {side} {symbols[0]}")
    return sig


def _moves(raw: str, lower: str) -> List[Signal]:
    """Stop-loss moves: one per 'COIN SL …' line, else one for the post."""
    per_line = []
    for m in LINE_MOVE.finditer(raw):
        symbol = normalize_symbol(m.group(1))
        if symbol:
            stop = to_float(m.group(2)) * (1000 if m.group(3) else 1)
            per_line.append(Signal("move_sl", symbol=symbol, new_stop=stop, notes=[f"new stop {stop:g}"]))
    if per_line:
        return per_line
    move = MOVE_VERB.search(raw)
    stop_level = _first(STOP_LEVEL, raw) or _first(LOOSE_STOP, raw)
    has_levels = bool(_first(r"\b(?:entry|inngang)\b", raw) or _take_profits(raw))
    if not (move or (stop_level and not has_levels and not _side(lower, False))):
        return []
    before_sl = re.search(r"([$#]?[A-Za-z][\w&]*)[ \t]+(?:sl|stop)\b", raw, re.IGNORECASE)
    symbol = _known_symbol(raw) or (normalize_symbol(before_sl.group(1)) if before_sl else None)
    sig = Signal("move_sl", symbol=symbol)
    tail = raw[move.start():] if move else raw
    if re.search(r"\b(entry|be|break[\s-]?even|breakeven|inngang|kjøpspris)\b", tail, re.IGNORECASE):
        sig.stop_to_entry = True
        sig.notes.append("stop to entry (break-even)")
        return [sig]
    number = re.search(r"(?:to|til|->|→|=|@|på|at|ved)\s*\$?(" + NUM + r")(k)?" + _NO_X_PCT, tail, re.IGNORECASE) \
        or stop_level
    if number:
        sig.new_stop = to_float(number.group(1)) * (1000 if number.group(2) else 1)
        sig.notes.append(f"new stop {sig.new_stop:g}")
        return [sig]
    return [Signal("info", notes=["mentions the stop but no new level found"])]


_SOLD_GAP = re.compile(r"[ \t]+(?:(?:min|mine|hele|denne|den|å|alt)[ \t]+)*$", re.IGNORECASE)


def parse_signals(text: str) -> List[Signal]:
    """All instructions in one post (e.g. two trades, or a close plus a new trade)."""
    if not text or not text.strip():
        return [Signal("info", notes=["empty post"])]
    raw = _join_thousands(text)
    for pattern, replacement in MULTIWORD:
        raw = re.sub(pattern, replacement, raw, flags=re.IGNORECASE)
    raw, other_platform = _liquid_only(raw)
    if not raw.strip():
        return [Signal("info", notes=["only about SizeProp/prop accounts, not Liquid"])]
    lower = raw.lower()
    out = _parse(raw, lower, other_platform)
    if other_platform:
        for sig in out:
            sig.notes.append("parts about SizeProp/prop accounts ignored (not Liquid)")
    return out


def _parse(raw: str, lower: str, other_platform: bool) -> List[Signal]:
    if re.search(r"\b(tp\s*\d|target\s*\d|mål\s*\d)\s*(hit|reached|nådd|tatt|done|✅)", lower):
        return [Signal("info", symbol=_symbol(raw), notes=["target-hit announcement; targets are tracked from prices"])]
    if re.search(r"\b(cancel\w*|avbryt\w*|kanseller\w*|ignore\s+(this|the)\s+(signal|trade|entry))\b", lower):
        return [Signal("cancel", symbol=_symbol(raw), notes=["cancel pending entry"])]

    heads = _headlines(raw)
    if any(h.symbols for h in heads):
        heads = [h for h in heads if h.symbols]   # "TESLA LONG 🔼 … longet mer nå" is one call
    closes = list(CLOSE_WORDS.finditer(raw))
    plans = [m for m in closes if _conditional(raw, m)]
    close = next((m for m in closes if m not in plans), None)
    notes = [f"'{m.group(0)}' is a plan ('if/when … I sell'), not a sell now" for m in plans[:1]]
    # "selger X long nå": the headline right after the sell word is what he sold.
    sold = [h for h in heads if close and h.symbol_first and h.start >= close.end()
            and _SOLD_GAP.fullmatch(raw[close.end():h.start] or " ")]
    heads = [h for h in heads if h not in sold]
    out: List[Signal] = []
    # A sell word before the headline ("SOLGT TESLA, LONGER NVIDIA") is its own instruction;
    # one after it is part of the trade's text.
    if close and (not heads or close.start() < heads[0].start):
        out.append(_close_signal(raw, lower, close, sold[0].symbols[0] if sold else None))

    if heads and STATUS.search(lower):   # "For øyeblikket long HOOD" = already in it
        if close and not out:            # "..., men SOLGT NVIDIA"
            out.append(_close_signal(raw, lower, close))
        h = heads[0]
        return out + [Signal("info", symbol=(h.symbols or [None])[0], side=h.side, notes=[
            f"status update: he is already {h.side} {', '.join(h.symbols)}, not a new buy"] + notes)]
    future = FUTURE_ENTRY.search(raw)
    count = sum(len(h.symbols) or 1 for h in heads)
    for h in heads:
        for sym in h.symbols or [None]:
            sig = _open_details(Signal("open", symbol=sym, side=h.side, add=h.add), raw, lower)
            what = f"{sym or 'his position'} {h.side}"
            if future:
                sig = Signal("info", symbol=sym, side=h.side, notes=[
                    f"plan for later ('{future.group(0).strip()}'): not bought now"])
            elif not (_has_trade_details(sig, raw, h.end) or h.verb or h.add):
                why = "looks like a joke/meme (😂, haha)" if JOKE.search(lower) else "chat, not a call"
                sig = Signal("info", symbol=sym, side=h.side, notes=[
                    f"{why}: no leverage, stop, levels or 🔼/🔽 next to '{what}', not bought"])
            elif h.add:
                sig.notes.insert(0, f"adds to {what}")
            if count > 1:
                sig.notes.append(f"one of {count} trades in the same post")
            out.append(sig)
    moves = [] if other_platform else _moves(raw, lower)
    if out:
        if any(s.action == "close" for s in out):   # "selger 25% og plasserer SL i profitt på $X"
            out += [m for m in moves if m.action == "move_sl"]
        return out
    if plans:
        return [Signal("info", symbol=_known_symbol(raw), notes=notes)]
    if moves:
        return moves
    prose = _prose_call(raw, lower)
    if prose:
        return [Signal("info", symbol=prose.symbol, side=prose.side, notes=[
            f"plan for later ('{future.group(0).strip()}'): not bought now"])] if future else [prose]

    # Classic labelled signal (entry/SL/TP) without a headline.
    stop_m = _first(STOP_LEVEL, raw)
    has_levels = bool(_first(r"\b(?:entry|inngang)\b", raw) or _take_profits(raw))
    if stop_m or has_levels:
        side, symbol = _side(lower, True), _symbol(raw)
        if side and symbol:
            return [_open_details(Signal("open", symbol=symbol, side=side), raw, lower)]
    side, symbol = _side(lower, False), _symbol(raw)
    missing = [n for n, v in (("direction (long/short) next to the coin", side and symbol), ) if not v]
    return [Signal("info", symbol=symbol, side=side, notes=["no trade: missing " + ", ".join(missing)])]


def parse_signal(text: str) -> Signal:
    """The first instruction in a post (see ``parse_signals`` for all of them)."""
    return parse_signals(text)[0]
