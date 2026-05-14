"""Shared connector primitives."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from urllib.parse import urlparse


@dataclass
class IngestedItem:
    """A normalized item produced by any connector before enrichment."""

    source: str
    text: str
    title: Optional[str] = None
    author: Optional[str] = None
    url: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    raw_payload: Optional[str] = None

    @property
    def source_domain(self) -> Optional[str]:
        if not self.url:
            return None
        try:
            return urlparse(self.url).netloc.lower().lstrip("www.")
        except Exception:
            return None

    def dedupe_key(self) -> str:
        """Stable hash used to de-duplicate the same story from multiple feeds."""
        basis = (self.url or "") + "|" + (self.title or self.text[:200])
        return hashlib.sha256(basis.encode("utf-8", errors="ignore")).hexdigest()[:32]


class Connector:
    """Abstract base for all ingestion connectors."""

    name: str = "base"

    def fetch(self) -> List[IngestedItem]:  # pragma: no cover - interface
        raise NotImplementedError
