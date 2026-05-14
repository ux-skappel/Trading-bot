"""RSS / Atom feed connector. Uses feedparser; degrades gracefully on errors."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

from ..config import get_settings
from .base import Connector, IngestedItem

logger = logging.getLogger(__name__)


class RSSConnector(Connector):
    """Reads a list of RSS/Atom feeds and produces IngestedItem records."""

    name = "rss"

    def __init__(self, feeds: Optional[List[str]] = None) -> None:
        self.feeds = feeds or get_settings().rss_feeds

    def fetch(self) -> List[IngestedItem]:
        try:
            import feedparser  # type: ignore
        except ImportError:
            logger.warning("feedparser not installed; returning no RSS items")
            return []

        items: List[IngestedItem] = []
        for url in self.feeds:
            try:
                parsed = feedparser.parse(url)
            except Exception as exc:  # network / parse errors
                logger.warning("RSS fetch failed for %s: %s", url, exc)
                continue

            for entry in parsed.entries[:50]:
                title = getattr(entry, "title", "")
                summary = getattr(entry, "summary", "") or getattr(entry, "description", "")
                link = getattr(entry, "link", None)
                author = getattr(entry, "author", None) or parsed.feed.get("title", url)
                ts_struct = getattr(entry, "published_parsed", None) or getattr(
                    entry, "updated_parsed", None
                )
                ts = datetime(*ts_struct[:6]) if ts_struct else datetime.utcnow()
                text = f"{title}\n{summary}".strip()
                if not text:
                    continue
                items.append(
                    IngestedItem(
                        source="rss",
                        title=title,
                        text=text,
                        author=author,
                        url=link,
                        timestamp=ts,
                    )
                )
        logger.info("RSS connector fetched %d items from %d feeds", len(items), len(self.feeds))
        return items
