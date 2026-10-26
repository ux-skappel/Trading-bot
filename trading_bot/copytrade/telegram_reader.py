"""Read the trader's Telegram posts, logged in as you (official API, read-only).

Nothing here sends, forwards or reacts to messages: it only reads one chat and
keeps one author's posts. Your login is kept in ``telegram.session`` in the
project folder; treat that file like a password and never share or commit it.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from ..config import get_settings
from ..storage.database import init_db
from .parser import parse_signal
from .store import save_message

logger = logging.getLogger(__name__)

SESSION = str(Path(__file__).resolve().parents[2] / "telegram")


def _client():
    from telethon import TelegramClient  # type: ignore

    s = get_settings()
    if not (s.telegram_api_id and s.telegram_api_hash):
        raise SystemExit(
            "Telegram is not set up. Create an app at https://my.telegram.org (API development "
            "tools) and put TELEGRAM_API_ID and TELEGRAM_API_HASH in your .env file."
        )
    return TelegramClient(SESSION, int(s.telegram_api_id), s.telegram_api_hash)


def _utc(dt: datetime) -> datetime:
    return dt.astimezone(timezone.utc).replace(tzinfo=None) if dt.tzinfo else dt


async def _resolve_chat(client, chat: str):
    if not chat:
        raise SystemExit("Set COPY_CHAT in .env (run `python -m trading_bot.main copy chats` to find it).")
    try:
        return await client.get_entity(int(chat) if chat.lstrip("-").isdigit() else chat)
    except Exception:
        async for dialog in client.iter_dialogs():
            if dialog.name and chat.lower() in dialog.name.lower():
                return dialog.entity
    raise SystemExit(f"Chat {chat!r} not found. Run `python -m trading_bot.main copy chats`.")


def _is_broadcast(entity) -> bool:
    return bool(getattr(entity, "broadcast", False))


async def _resolve_author(client, entity, author: str):
    if _is_broadcast(entity) or not author:
        return None
    return await client.get_entity(int(author) if author.lstrip("-").isdigit() else author)


def _sender_name(msg) -> Optional[str]:
    sender = getattr(msg, "sender", None)
    if sender is None:
        return None
    username = getattr(sender, "username", None)
    name = " ".join(p for p in (getattr(sender, "first_name", None), getattr(sender, "last_name", None)) if p)
    return f"@{username}" if username else (name or getattr(sender, "title", None))


def _save(chat_id, msg) -> str:
    reply = msg.reply_to.reply_to_msg_id if getattr(msg, "reply_to", None) else None
    return save_message(str(chat_id), msg.id, _utc(msg.date), msg.message or "", _sender_name(msg), reply)


async def _login(client) -> None:
    me = await client.get_me()
    print(f"Logged in to Telegram as {me.first_name} (@{me.username}). Session saved; keep 'telegram.session' private.")


async def _list_chats(client, limit: int = 200) -> None:
    print(f"{'ID':>16}  TYPE       NAME")
    async for d in client.iter_dialogs(limit=limit):
        kind = "channel" if _is_broadcast(d.entity) else "group" if d.is_group else "user"
        print(f"{d.id:>16}  {kind:<9}  {d.name}")


async def _list_authors(client, chat: str, limit: int = 500) -> None:
    entity = await _resolve_chat(client, chat)
    tally: dict = {}
    async for msg in client.iter_messages(entity, limit=limit):
        if msg.sender_id is None:
            continue
        key = (msg.sender_id, _sender_name(msg))
        tally[key] = tally.get(key, 0) + 1
    print(f"Most active in the last {limit} messages:")
    for (sender_id, name), count in sorted(tally.items(), key=lambda kv: -kv[1])[:25]:
        print(f"{count:>5}  {name or '?':<30} id {sender_id}")


async def _backfill(client, chat: str, author: str, days: int) -> None:
    entity = await _resolve_chat(client, chat)
    who = await _resolve_author(client, entity, author)
    since = datetime.utcnow() - timedelta(days=days)
    counts = {"new": 0, "edited": 0, "same": 0}
    async for msg in client.iter_messages(entity, from_user=who):
        if _utc(msg.date) < since:
            break
        if not msg.message:
            continue
        counts[_save(entity.id, msg)] += 1
    print(f"Fetched {sum(counts.values())} posts from the last {days} days "
          f"({counts['new']} new, {counts['edited']} edited).")


async def _listen(client, chat: str, author: str) -> None:
    from telethon import events  # type: ignore

    entity = await _resolve_chat(client, chat)
    who = await _resolve_author(client, entity, author)
    kwargs = {"chats": entity}
    if who is not None:
        kwargs["from_users"] = who

    async def on_message(event) -> None:
        msg = event.message
        if not msg.message:
            return
        status = _save(entity.id, msg)
        signal = parse_signal(msg.message)
        stamp = _utc(msg.date).strftime("%Y-%m-%d %H:%M")
        print(f"[{stamp} UTC] {status}: {signal.summary()}")

    client.add_event_handler(on_message, events.NewMessage(**kwargs))
    client.add_event_handler(on_message, events.MessageEdited(**kwargs))
    print("Listening (read-only). New posts are saved and paper-copied in the dashboard. Ctrl+C to stop.")
    await client.run_until_disconnected()


def run(command: str, days: int = 90, limit: int = 500) -> None:
    """Entry point for the CLI: login | chats | authors | backfill | listen."""
    init_db()
    s = get_settings()
    client = _client()
    jobs = {
        "login": lambda: _login(client),
        "chats": lambda: _list_chats(client),
        "authors": lambda: _list_authors(client, s.copy_chat, limit),
        "backfill": lambda: _backfill(client, s.copy_chat, s.copy_author, days),
        "listen": lambda: _listen(client, s.copy_chat, s.copy_author),
    }
    async def main() -> None:
        await client.start(phone=s.telegram_phone or None)   # asks for the login code the first time
        try:
            await jobs[command]()
        finally:
            await client.disconnect()

    client.loop.run_until_complete(main())
