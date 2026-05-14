"""Database session and initialization helpers."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from ..config import get_settings
from .models import Base, HighImpactAccount, Source

logger = logging.getLogger(__name__)

_settings = get_settings()
engine = create_engine(_settings.database_url, future=True, echo=False)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_db() -> None:
    """Create tables and seed registry data on first run."""
    Base.metadata.create_all(engine)
    _seed_high_impact_accounts()
    _seed_sources()
    logger.info("Database initialized at %s", _settings.database_url)


def _seed_high_impact_accounts() -> None:
    """Sync the high-impact account watchlist from config into the DB."""
    with session_scope() as db:
        existing = {row.username for row in db.query(HighImpactAccount).all()}
        for username, meta in _settings.high_impact_accounts.items():
            if username in existing:
                continue
            db.add(
                HighImpactAccount(
                    username=username,
                    display_name=meta.get("name", username),
                    category=meta.get("category", "other"),
                    credibility=float(meta.get("credibility", 0.5)),
                    official=bool(meta.get("official", False)),
                    active=True,
                )
            )


def _seed_sources() -> None:
    """Seed source credibility registry from config."""
    with session_scope() as db:
        existing = {row.domain for row in db.query(Source).all()}
        for domain, cred in _settings.source_credibility.items():
            if domain in existing:
                continue
            db.add(Source(domain=domain, credibility=float(cred)))


@contextmanager
def session_scope() -> Iterator[Session]:
    """Provide a transactional scope for a series of operations."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
