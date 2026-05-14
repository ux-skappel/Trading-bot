"""Reddit connector — official API via praw. No-op if credentials are absent."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

from ..config import get_settings
from .base import Connector, IngestedItem

logger = logging.getLogger(__name__)


class RedditConnector(Connector):
    """Pulls newest submissions from configured subreddits."""

    name = "reddit"

    def __init__(self, subreddits: Optional[List[str]] = None, limit: int = 25) -> None:
        s = get_settings()
        self.client_id = s.reddit_client_id
        self.client_secret = s.reddit_client_secret
        self.user_agent = s.reddit_user_agent
        self.subreddits = subreddits or s.reddit_subreddits
        self.limit = limit

    def _client(self):
        if not (self.client_id and self.client_secret):
            return None
        try:
            import praw  # type: ignore
        except ImportError:
            logger.warning("praw not installed; Reddit connector disabled")
            return None
        return praw.Reddit(
            client_id=self.client_id,
            client_secret=self.client_secret,
            user_agent=self.user_agent,
            check_for_async=False,
        )

    def fetch(self) -> List[IngestedItem]:
        reddit = self._client()
        if reddit is None:
            logger.info("Reddit connector skipped (no credentials).")
            return []

        items: List[IngestedItem] = []
        for sub in self.subreddits:
            try:
                for s in reddit.subreddit(sub).new(limit=self.limit):
                    text = (s.title or "") + "\n" + (s.selftext or "")
                    items.append(
                        IngestedItem(
                            source="reddit",
                            title=s.title,
                            text=text.strip(),
                            author=f"u/{s.author}" if s.author else None,
                            url=f"https://reddit.com{s.permalink}",
                            timestamp=datetime.utcfromtimestamp(s.created_utc),
                        )
                    )
            except Exception as exc:
                logger.warning("Reddit fetch failed for r/%s: %s", sub, exc)
        logger.info("Reddit connector fetched %d items", len(items))
        return items
