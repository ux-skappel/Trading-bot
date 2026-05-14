from trading_bot.analysis.high_impact import classify_high_impact_post


def test_unknown_author_not_flagged():
    r = classify_high_impact_post("randomuser", "tariffs on china")
    assert r.is_high_impact is False


def test_trump_tariffs_detected_as_watch_only():
    r = classify_high_impact_post("realDonaldTrump", "Massive tariffs on China incoming!")
    assert r.is_high_impact
    assert "tariffs" in r.detected_topics
    assert r.bias == -1
    assert r.watch_only is True   # politician, non-official account
    assert r.requires_extra_confirmation is True


def test_fed_dovish_post():
    r = classify_high_impact_post("federalreserve", "FOMC signals rate cut")
    assert r.is_high_impact
    assert "interest_rates" in r.detected_topics
    assert r.official_account is True


def test_musk_joke_dampens_impact():
    r = classify_high_impact_post("elonmusk", "Doge to the moon 🚀 lol")
    assert r.is_high_impact
    assert r.tone in {"joke", "neutral"}
    assert r.expected_impact <= 0.4


def test_sec_crypto_regulation():
    r = classify_high_impact_post("SECGov", "SEC issues new crypto regulation guidance")
    assert "crypto_regulation" in r.detected_topics
    assert r.official_account is True
    assert "BTC" in r.affected_crypto
