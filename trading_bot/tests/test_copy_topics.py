from types import SimpleNamespace as NS

import pytest

from trading_bot.copytrade.telegram_reader import GENERAL_TOPIC, _reply_id, _says_sell, _topic_of


@pytest.mark.parametrize("caption,read", [
    ("SELGER HELE NÅ 🔼🤑", True),
    ("Solgt halvparten", True),
    ("SL i profitt på $81.2K", True),
    ("GOOGLE LONG 🔼 Bruker 20X giring og SL på $320", False),
    ("Se så fint det gikk 🤑", False),
    ("", False),
    (None, False),
])
def test_only_pictures_on_sell_posts_are_read(caption, read):
    assert _says_sell(caption) is read


def reply(msg_id=None, top_id=None, forum_topic=False):
    return NS(reply_to=NS(reply_to_msg_id=msg_id, reply_to_top_id=top_id, forum_topic=forum_topic))


def test_message_without_reply_is_in_the_general_topic():
    msg = NS(reply_to=None)
    assert _topic_of(msg) == GENERAL_TOPIC and _reply_id(msg) is None


def test_top_level_post_in_a_topic_is_not_a_real_reply():
    msg = reply(msg_id=55, forum_topic=True)          # Telegram points it at the topic root
    assert _topic_of(msg) == 55
    assert _reply_id(msg) is None


def test_reply_inside_a_topic_keeps_both():
    msg = reply(msg_id=1234, top_id=55, forum_topic=True)
    assert _topic_of(msg) == 55
    assert _reply_id(msg) == 1234


def test_plain_reply_in_a_group_without_topics():
    msg = reply(msg_id=77)
    assert _topic_of(msg) == GENERAL_TOPIC
    assert _reply_id(msg) == 77
