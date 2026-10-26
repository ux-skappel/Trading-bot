import json
from datetime import datetime

from trading_bot.copytrade.store import export_authors, import_telegram_export, load_posts
from trading_bot.storage.database import init_db


def _export(tmp_path):
    data = {"name": "Signals", "id": 4242, "messages": [
        {"id": 1, "type": "message", "date": "2026-01-01T13:00:00", "date_unixtime": "1767268800",
         "from": "Trader Ola", "from_id": "user9", "text": "LONG $SOL entry 100 SL 95"},
        {"id": 2, "type": "message", "date": "2026-01-01T13:05:00", "date_unixtime": "1767269100",
         "from": "Someone", "from_id": "user1", "text": "great call"},
        {"id": 3, "type": "message", "date": "2026-01-01T14:00:00", "date_unixtime": "1767272400",
         "from": "Trader Ola", "from_id": "user9", "reply_to_message_id": 1,
         "text": ["Move ", {"type": "bold", "text": "SL"}, " to entry"]},
        {"id": 4, "type": "service", "date": "2026-01-01T15:00:00", "action": "pin_message"},
    ]}
    path = tmp_path / "result.json"
    path.write_text(json.dumps(data))
    return str(path)


def test_import_keeps_only_his_posts_with_replies_and_utc_times(tmp_path):
    init_db()
    path = _export(tmp_path)
    assert export_authors(path)[0] == ("Trader Ola", "user9", 2)
    counts = import_telegram_export(path, author="trader ola")
    assert counts["new"] == 2 and counts["skipped"] == 2
    posts = [p for p in load_posts("4242")]
    assert [p.message_id for p in posts] == [1, 3]
    assert posts[1].text == "Move SL to entry" and posts[1].reply_to == 1
    assert posts[0].time == datetime(2026, 1, 1, 12, 0)          # from date_unixtime (UTC)

    again = import_telegram_export(path, author="user9")
    assert again["new"] == 0 and again["same"] == 2
