"""Save and load the trader's posts (SQLite), incl. Telegram Desktop exports."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Iterable, List, Optional

from sqlalchemy import select

from ..storage.database import init_db, session_scope
from ..storage.models import CopyMessage
from .engine import Post


def save_message(chat_id: str, message_id: int, timestamp: datetime, text: str,
                 author: Optional[str] = None, reply_to: Optional[int] = None) -> str:
    """Insert or update one post. Returns 'new', 'edited' or 'same'."""
    with session_scope() as db:
        row = db.scalars(select(CopyMessage).where(
            CopyMessage.chat_id == str(chat_id), CopyMessage.message_id == message_id)).first()
        if row is None:
            db.add(CopyMessage(chat_id=str(chat_id), message_id=message_id, author=author,
                               timestamp=timestamp, text=text, reply_to=reply_to))
            return "new"
        if row.text != text:
            row.text, row.edited = text, True
            return "edited"
        return "same"


def load_posts(chat_id: Optional[str] = None) -> List[Post]:
    """All stored posts (optionally one chat), oldest first."""
    init_db()
    with session_scope() as db:
        query = select(CopyMessage).order_by(CopyMessage.timestamp, CopyMessage.message_id)
        if chat_id:
            query = query.where(CopyMessage.chat_id == str(chat_id))
        return [Post(r.message_id, r.timestamp, r.text, r.reply_to) for r in db.scalars(query)]


def _export_text(text) -> str:
    if isinstance(text, str):
        return text
    return "".join(part if isinstance(part, str) else part.get("text", "") for part in text or [])


def import_telegram_export(path: str, author: Optional[str] = None) -> dict:
    """Import a Telegram Desktop export (result.json), keeping one author's posts.

    ``author`` matches the sender name or id shown in the export (case-insensitive,
    e.g. "Ola Trader" or "user12345"). Leave it empty for a channel.
    """
    init_db()
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    chat_id = str(data.get("id", data.get("name", "export")))
    wanted = author.lower().lstrip("@") if author else None
    counts = {"new": 0, "edited": 0, "same": 0, "skipped": 0}
    for msg in data.get("messages", []):
        text = _export_text(msg.get("text")).strip()
        sender = str(msg.get("from") or "")
        sender_id = str(msg.get("from_id") or "")
        if msg.get("type") != "message" or not text or \
                (wanted and wanted not in (sender.lower(), sender_id.lower())):
            counts["skipped"] += 1
            continue
        if msg.get("date_unixtime"):
            stamp = datetime.utcfromtimestamp(int(msg["date_unixtime"]))
        else:
            stamp = datetime.fromisoformat(msg["date"])
        status = save_message(chat_id, int(msg["id"]), stamp, text, sender or None,
                              msg.get("reply_to_message_id"))
        counts[status] += 1
    counts["chat_id"] = chat_id
    return counts


def export_authors(path: str) -> List[tuple]:
    """(sender, sender id, number of posts) in a Telegram Desktop export, busiest first."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    tally: dict = {}
    for msg in data.get("messages", []):
        if msg.get("type") == "message":
            key = (str(msg.get("from") or ""), str(msg.get("from_id") or ""))
            tally[key] = tally.get(key, 0) + 1
    return sorted(((n, i, c) for (n, i), c in tally.items()), key=lambda r: -r[2])
