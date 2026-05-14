from trading_bot.analysis.entity_extraction import extract_entities


def test_finds_us_ticker():
    e = extract_entities("Strong earnings report from $AAPL today")
    assert "AAPL" in e.tickers


def test_finds_norwegian_ticker():
    e = extract_entities("EQNR.OL announces dividend")
    assert "EQNR.OL" in e.tickers


def test_finds_crypto_symbols():
    e = extract_entities("Bitcoin and Ethereum both rally; DOGE flat")
    assert "BTC" in e.crypto
    assert "ETH" in e.crypto
    assert "DOGE" in e.crypto


def test_blacklist_filters_common_words():
    e = extract_entities("I will go to USA on EU summit")
    assert "I" not in e.tickers
    assert "USA" not in e.tickers
    assert "EU" not in e.tickers
