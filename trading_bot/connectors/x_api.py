"""X / Twitter connector — official API v2 only.

If no bearer token is configured, this connector returns no items. It never
scrapes web pages or bypasses the platform terms of service.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

from ..config import get_settings
from .base import Connector, IngestedItem

logger = logging.getLogger(__name__)


class XConnector(Connector):
    """Pulls recent posts from configured high-impact accounts via the X API."""

    name = "x"

    def __init__(self, usernames: Optional[List[str]] = None) -> None:
        settings = get_settings()
        self.bearer_token = settings.x_bearer_token
        self.usernames = usernames or list(settings.high_impact_accounts.keys())

    def _client(self):
        if not self.bearer_token:
            return None
        try:
            import tweepy  # type: ignore
        except ImportError:
            logger.warning("tweepy not installed; X connector disabled")
            return None
        return tweepy.Client(bearer_token=self.bearer_token, wait_on_rate_limit=False)

    def fetch(self) -> List[IngestedItem]:
        client = self._client()
        if client is None:
            logger.info("X connector skipped (no bearer token).")
            return []

        items: List[IngestedItem] = []
        for username in self.usernames:
            try:
                user = client.get_user(username=username)
                if not user.data:
                    continue
                tweets = client.get_users_tweets(
                    id=user.data.id,
                    max_results=10,
                    tweet_fields=["created_at", "text", "entities"],
                )
                for tw in tweets.data or []:
                    items.append(
                        IngestedItem(
                            source="x",
                            text=tw.text,
                            author=username,
                            url=f"https://x.com/{username}/status/{tw.id}",
                            timestamp=tw.created_at or datetime.utcnow(),
                        )
                    )
            except Exception as exc:  # rate limit / auth / network
                logger.warning("X fetch failed for @%s: %s", username, exc)
        logger.info("X connector fetched %d items", len(items))
        return items
