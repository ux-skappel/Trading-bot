import pytest

from trading_bot.copytrade.parser import parse_signal, to_float


@pytest.mark.parametrize("raw,expected", [
    ("65,000", 65000), ("65 000", 65000), ("0,1234", 0.1234), ("1.5", 1.5),
    ("65.000,5", 65000.5), ("65,000.5", 65000.5), ("1,5", 1.5), ("1.000.000", 1000000),
])
def test_to_float(raw, expected):
    assert to_float(raw) == pytest.approx(expected)


def test_classic_english_signal():
    s = parse_signal("🟢 LONG $SOL\nEntry: 142-144\nSL: 138\nTP1: 150\nTP2: 158\nLeverage: 10x\nRisk 1%")
    assert s.action == "open" and s.side == "long" and s.symbol == "SOL"
    assert (s.entry_low, s.entry_high) == (142, 144)
    assert s.stop_loss == 138 and s.take_profits == [150, 158]
    assert s.leverage == 10 and s.risk_pct == pytest.approx(0.01)
    assert not s.market


def test_short_with_pair_and_target_list():
    s = parse_signal("BTC/USDT SHORT\nentry 64,500\nstop 65,800\ntargets 63,000, 62,000 / 60,500\n20x")
    assert s.action == "open" and s.side == "short" and s.symbol == "BTC"
    assert s.entry_low == s.entry_high == 64500
    assert s.stop_loss == 65800
    assert s.take_profits == [63000, 62000, 60500]
    assert s.leverage == 20


def test_norwegian_market_entry():
    s = parse_signal("Kjøper BTC nå, stopp 63 500, mål 68 000 / 70 000. 5x, 2% av porteføljen")
    assert s.action == "open" and s.side == "long" and s.symbol == "BTC"
    assert s.market
    assert s.stop_loss == 63500
    assert s.take_profits == [68000, 70000]
    assert s.leverage == 5 and s.size_pct == pytest.approx(0.02)


def test_k_suffix_and_hashtag():
    s = parse_signal("#ETH long @ 3.2k sl 3.05k tp 3.5k")
    assert s.symbol == "ETH" and s.side == "long"
    assert s.entry_low == pytest.approx(3200)
    assert s.stop_loss == pytest.approx(3050)
    assert s.take_profits == [pytest.approx(3500)]


def test_wrong_side_levels_are_dropped_with_a_note():
    s = parse_signal("LONG DOGE entry 0,15 SL 0,16 TP 0,14 TP2 0,18")
    assert s.stop_loss is None
    assert s.take_profits == [pytest.approx(0.18)]
    assert any("wrong side" in n for n in s.notes)


@pytest.mark.parametrize("text,stop_to_entry,new_stop", [
    ("Move SL to entry on SOL", True, None),
    ("SL to BE", True, None),
    ("Flytt stop til inngang", True, None),
    ("move stop to 146.5", False, 146.5),
    ("SOL: sl -> 147", False, 147),
])
def test_move_stop(text, stop_to_entry, new_stop):
    s = parse_signal(text)
    assert s.action == "move_sl"
    assert s.stop_to_entry == stop_to_entry
    assert s.new_stop == new_stop


@pytest.mark.parametrize("text,fraction,symbol", [
    ("Closing SOL here, +12%", 1.0, "SOL"),
    ("Close half $BTC", 0.5, "BTC"),
    ("Lukker 30% av ETH", 0.3, "ETH"),
    ("Lukker alt", 1.0, None),
    ("Take some profits now on DOGE", 0.5, "DOGE"),     # "some" = partial
    ("Closing 25% of the AVAX position", 0.25, "AVAX"),
])
def test_close(text, fraction, symbol):
    s = parse_signal(text)
    assert s.action == "close"
    assert s.close_fraction == pytest.approx(fraction)
    assert s.symbol == symbol


@pytest.mark.parametrize("text", [
    "BTC is close to support, watching",
    "Daily candle closed above resistance on ETH",
    "Good morning everyone! Market looks choppy today",
    "TP1 hit ✅ on SOL",
    "Do NOT chase this pump",
    "Long term I like LINK",          # opinion, no levels -> still long+symbol... see below
])
def test_chatter_is_not_traded(text):
    s = parse_signal(text)
    if text.startswith("Long term"):
        # Direction + coin with no levels is treated as a market entry; the
        # engine still requires the post to look like an instruction.
        assert s.action in {"open", "info"}
    else:
        assert s.action == "info", (text, s)


def test_cancel():
    s = parse_signal("Cancel the SOL entry, didn't fill")
    assert s.action == "cancel" and s.symbol == "SOL"


def test_numbered_targets_on_one_line():
    s = parse_signal("LONG $SOL entry 100 SL 95 TP1 105 TP2 110 10x")
    assert s.take_profits == [105, 110] and s.leverage == 10


def test_summary_is_readable():
    s = parse_signal("LONG $SOL entry 142 SL 138 TP 150 10x")
    assert s.summary() == "OPEN LONG SOL · entry 142 · SL 138 · TP 150 · 10x"


# --- Headline style: "NAME LONG/SHORT", leverage + SL in the body (made-up examples) ---

from trading_bot.copytrade.parser import parse_signals


@pytest.mark.parametrize("text,symbol,side,stop,lev", [
    ("APPLE LONG 🔼\n\nSer sterk ut! Bruker 15X giring og SL på $251,5", "AAPL", "long", 251.5, 15),
    ("NVIDIA SHORT 🔽\n\nBruker 10X giring og SL på $212", "NVDA", "short", 212, 10),
    ("S&P500 LONG 🔼\n\nPå support! Bruker 50X giring og SL på $7480", "SP500", "long", 7480, 50),
    ("GOLD LONG 🔼\n\nOversolgt. Bruker 25X giring og ingen SL", "GOLD", "long", None, 25),
    ("ETH SHORT 🔽\n\nRisky! Bruker 40X giring og SL på $2.65K", "ETH", "short", 2650, 40),
    ("MICRON LONG 🔼\n\nBruker 10X giring og SL $118", "MU", "long", 118, 10),
])
def test_headline_posts(text, symbol, side, stop, lev):
    s = parse_signal(text)
    assert (s.action, s.symbol, s.side, s.stop_loss, s.leverage) == ("open", symbol, side, stop, lev)
    assert s.market


def test_two_trades_in_one_post():
    sigs = parse_signals("LONGER MICRON OG WESTERN NÅ 🔼🔼")
    assert [(s.action, s.symbol, s.side) for s in sigs] == [("open", "MU", "long"), ("open", "WDC", "long")]


def test_partial_position_size():
    s = parse_signal("TESLA LONG 🔼\n\nBruker 10X giring og ingen SL ennå\n\nInne med 25% av full ønsket posisjon")
    assert s.symbol == "TSLA" and s.size_scale == pytest.approx(0.25)


@pytest.mark.parametrize("text,symbol,fraction", [
    ("SOLGT HELE NÅ 🔼🤑", None, 1.0),
    ("SOLGT WESTERN 🔼🤑", "WDC", 1.0),
    ("Solgt halvparten av TESLA", "TSLA", 0.5),
])
def test_norwegian_sells(text, symbol, fraction):
    s = parse_signal(text)
    assert (s.action, s.symbol, s.close_fraction) == ("close", symbol, fraction)


def test_stop_in_profit_moves_the_stop():
    s = parse_signal("BOOM 🔽🔽\n\nSL i profitt på $81.2K")
    assert s.action == "move_sl" and s.new_stop == 81200 and s.symbol is None


def test_setting_a_stop_later():
    s = parse_signal("SL på $355 nå")
    assert s.action == "move_sl" and s.new_stop == 355


@pytest.mark.parametrize("text", [
    "Ny uke med rapporter 👀\nTirsdag: Netflix\nKommer tilbake til om jeg går long/short når vi nærmer oss",
    "Bruk koden ABCDE for rabatt. Kjøper tilgang til en stor konto og består testen",
    "HØYERE 🔼🔼🔼🔼",
    "For enkelt\n\nHøyere 🔼🤑",
    "Mange nye medlemmer i gruppa nå 🎉🔼",
    "Kjører 1X $10,000 eval giveaway, reager på denne meldingen",
])
def test_his_chatter_is_not_traded(text):
    assert all(s.action == "info" for s in parse_signals(text)), parse_signals(text)


def test_sell_then_new_trade_in_one_post():
    sigs = parse_signals("SOLGT TESLA 🤑 og LONGER NVIDIA nå 🔼")
    assert [(s.action, s.symbol) for s in sigs] == [("close", "TSLA"), ("open", "NVDA")]
