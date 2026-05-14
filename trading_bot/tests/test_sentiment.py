from trading_bot.analysis.sentiment import is_rumor, score_sentiment


def test_positive():
    assert score_sentiment("Earnings beat estimates, guidance raised") > 0


def test_negative():
    assert score_sentiment("Massive lawsuit, fraud probe, downgrade") < 0


def test_neutral():
    assert score_sentiment("The company filed paperwork today") == 0.0


def test_rumor_flag():
    assert is_rumor("Reportedly, sources say a deal may be near")
    assert not is_rumor("Company confirmed the acquisition")
