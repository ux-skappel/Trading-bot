"""Read the trader's Telegram posts, logged in as you (official API, read-only).

Nothing here sends, forwards or reacts to messages: it only reads one chat
(optionally one topic in a group with topics) and keeps one author's posts.
Your login is kept in ``telegram.session`` in the project folder; treat that
file like a password and never share or commit it.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

from ..config import get_settings
from ..storage.database import init_db
from .parser import parse_signal
from .store import delete_chat_posts, save_message

logger = logging.getLogger(__name__)

SESSION = str(Path(__file__).resolve().parents[2] / "telegram")
GENERAL_TOPIC = 1


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


async def _forum_topics(client, entity) -> list:
    """(id, title) for every topic, or [] if the group has no topics."""
    if not getattr(entity, "forum", False):
        return []
    from telethon import functions  # type: ignore

    if hasattr(functions.messages, "GetForumTopicsRequest"):
        request = functions.messages.GetForumTopicsRequest(
            peer=entity, offset_date=None, offset_id=0, offset_topic=0, limit=100)
    else:  # older Telethon
        request = functions.channels.GetForumTopicsRequest(
            channel=entity, offset_date=None, offset_id=0, offset_topic=0, limit=100)
    result = await client(request)
    return [(t.id, getattr(t, "title", "?")) for t in result.topics]


async def _resolve_topics(client, entity, topics: str) -> List[int]:
    """Topic ids from COPY_TOPIC: ids or titles, comma-separated ([] = whole chat)."""
    wanted = [t.strip() for t in (topics or "").split(",") if t.strip()]
    if not wanted:
        return []
    known = await _forum_topics(client, entity)
    ids = []
    for topic in wanted:
        if topic.isdigit():
            ids.append(int(topic))
            continue
        match = [tid for tid, title in known if topic.lower() in title.lower()]
        if not match:
            raise SystemExit(f"Topic {topic!r} not found. Run `python -m trading_bot.main copy topics`.")
        ids.append(match[0])
    return ids


def _topic_of(msg) -> int:
    """Which topic a message is in (the General topic when it has none)."""
    rt = getattr(msg, "reply_to", None)
    if rt is None:
        return GENERAL_TOPIC
    if rt.reply_to_top_id:
        return rt.reply_to_top_id
    return rt.reply_to_msg_id if rt.forum_topic else GENERAL_TOPIC


def _reply_id(msg) -> Optional[int]:
    """The message this one really replies to (not just the topic it sits in)."""
    rt = getattr(msg, "reply_to", None)
    if rt is None or (rt.forum_topic and not rt.reply_to_top_id):
        return None
    return rt.reply_to_msg_id


def _sender_name(msg) -> Optional[str]:
    sender = getattr(msg, "sender", None)
    if sender is None:
        return None
    username = getattr(sender, "username", None)
    name = " ".join(p for p in (getattr(sender, "first_name", None), getattr(sender, "last_name", None)) if p)
    return f"@{username}" if username else (name or getattr(sender, "title", None))


def _save(chat_id, msg) -> str:
    return save_message(str(chat_id), msg.id, _utc(msg.date), msg.message or "",
                        _sender_name(msg), _reply_id(msg))


async def _login(client) -> None:
    me = await client.get_me()
    handle = f" (@{me.username})" if me.username else ""
    print(f"Logged in to Telegram as {me.first_name}{handle}. Session saved; keep 'telegram.session' private.")


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


async def _list_topics(client, chat: str, author: str, limit: int = 1000) -> None:
    entity = await _resolve_chat(client, chat)
    topics = await _forum_topics(client, entity)
    if not topics:
        print("This chat has no topics: leave COPY_TOPIC empty.")
        return
    who = await _resolve_author(client, entity, author)
    his, images = {}, {}
    async for msg in client.iter_messages(entity, limit=limit):
        if who is not None and msg.sender_id != who.id:
            continue
        topic = _topic_of(msg)
        his[topic] = his.get(topic, 0) + 1
        if msg.media and not msg.message:
            images[topic] = images.get(topic, 0) + 1
    whose = "his" if who is not None else "all"
    print(f"Topics ({whose} posts among the last {limit} messages):")
    print(f"{'ID':>8}  {'POSTS':>5}  {'IMAGES ONLY':>11}  TITLE")
    for topic_id, title in sorted(topics, key=lambda t: -his.get(t[0], 0)):
        print(f"{topic_id:>8}  {his.get(topic_id, 0):>5}  {images.get(topic_id, 0):>11}  {title}")


async def _backfill(client, chat: str, author: str, topics: str, days: int, fresh: bool) -> None:
    entity = await _resolve_chat(client, chat)
    who = await _resolve_author(client, entity, author)
    topic_ids = await _resolve_topics(client, entity, topics)
    if fresh:
        removed = delete_chat_posts(str(entity.id))
        print(f"Removed {removed} previously fetched posts.")
    since = datetime.utcnow() - timedelta(days=days)
    for topic_id in topic_ids or [None]:
        counts = {"new": 0, "edited": 0, "same": 0}
        images_only = 0
        if topic_id is None:
            messages = client.iter_messages(entity, from_user=who)
        elif topic_id == GENERAL_TOPIC:
            messages = client.iter_messages(entity, from_user=who)   # General has no thread to open
        else:
            messages = client.iter_messages(entity, reply_to=topic_id)
        async for msg in messages:
            if _utc(msg.date) < since:
                break
            if who is not None and msg.sender_id != who.id:
                continue
            if topic_id == GENERAL_TOPIC and _topic_of(msg) != GENERAL_TOPIC:
                continue
            if not msg.message:
                images_only += msg.media is not None
                continue
            counts[_save(entity.id, msg)] += 1
        where = f" in topic {topic_id}" if topic_id is not None else ""
        extra = f", {images_only} images without text skipped" if images_only else ""
        print(f"Fetched {sum(counts.values())} posts{where} from the last {days} days "
              f"({counts['new']} new{extra}).")


async def _listen(client, chat: str, author: str, topic: str) -> None:
    from telethon import events  # type: ignore

    print("Catching up on the last 3 days first (in case the computer was off or asleep)...")
    await _backfill(client, chat, author, topic, days=3, fresh=False)
    entity = await _resolve_chat(client, chat)
    who = await _resolve_author(client, entity, author)
    topic_ids = await _resolve_topics(client, entity, topic)
    kwargs = {"chats": entity}
    if who is not None:
        kwargs["from_users"] = who

    async def on_message(event) -> None:
        msg = event.message
        if topic_ids and _topic_of(msg) not in topic_ids:
            return
        stamp = _utc(msg.date).strftime("%Y-%m-%d %H:%M")
        if not msg.message:
            if msg.media:
                print(f"[{stamp} UTC] image without text: skipped")
            return
        status = _save(entity.id, msg)
        print(f"[{stamp} UTC] {status}: {parse_signal(msg.message).summary()}")

    client.add_event_handler(on_message, events.NewMessage(**kwargs))
    client.add_event_handler(on_message, events.MessageEdited(**kwargs))
    where = f", topics {', '.join(map(str, topic_ids))}" if topic_ids else ""
    print(f"Listening (read-only{where}). New posts are saved and paper-copied in the dashboard. "
          "Ctrl+C to stop.")
    await client.run_until_disconnected()


def run(command: str, days: int = 90, limit: int = 500, fresh: bool = False) -> None:
    """Entry point for the CLI: login | chats | authors | topics | backfill | listen."""
    init_db()
    s = get_settings()
    client = _client()
    jobs = {
        "login": lambda: _login(client),
        "chats": lambda: _list_chats(client),
        "authors": lambda: _list_authors(client, s.copy_chat, limit),
        "topics": lambda: _list_topics(client, s.copy_chat, s.copy_author, max(limit, 1000)),
        "backfill": lambda: _backfill(client, s.copy_chat, s.copy_author, s.copy_topic, days, fresh),
        "listen": lambda: _listen(client, s.copy_chat, s.copy_author, s.copy_topic),
    }

    async def main() -> None:
        await client.start(phone=s.telegram_phone or None)   # asks for the login code the first time
        try:
            await jobs[command]()
        finally:
            await client.disconnect()

    client.loop.run_until_complete(main())
