"""SQLAlchemy ORM models for the trading bot."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""


class RawEvent(Base):
    """Raw, unprocessed ingestion record."""

    __tablename__ = "raw_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    source: Mapped[str] = mapped_column(String(64), index=True)
    author: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    url: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    title: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    raw_payload: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    dedupe_key: Mapped[str] = mapped_column(String(64), index=True, unique=True)


class NormalizedEvent(Base):
    """Cleaned + enriched event ready for scoring."""

    __tablename__ = "normalized_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    raw_event_id: Mapped[Optional[int]] = mapped_column(ForeignKey("raw_events.id"))
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    source: Mapped[str] = mapped_column(String(64), index=True)
    source_domain: Mapped[Optional[str]] = mapped_column(String(128))
    author: Mapped[Optional[str]] = mapped_column(String(128), index=True)
    text: Mapped[str] = mapped_column(Text)
    url: Mapped[Optional[str]] = mapped_column(String(1024))
    companies: Mapped[Optional[str]] = mapped_column(Text)   # CSV
    tickers: Mapped[Optional[str]] = mapped_column(Text)     # CSV
    crypto_symbols: Mapped[Optional[str]] = mapped_column(Text)  # CSV
    event_type: Mapped[Optional[str]] = mapped_column(String(64))
    sentiment: Mapped[float] = mapped_column(Float, default=0.0)
    is_rumor: Mapped[bool] = mapped_column(Boolean, default=False)
    is_high_impact_account: Mapped[bool] = mapped_column(Boolean, default=False)
    high_impact_topics: Mapped[Optional[str]] = mapped_column(Text)  # CSV
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    dedupe_key: Mapped[str] = mapped_column(String(64), index=True)


class Asset(Base):
    """An asset (stock or crypto) tracked by the bot."""

    __tablename__ = "assets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    name: Mapped[Optional[str]] = mapped_column(String(255))
    asset_class: Mapped[str] = mapped_column(String(16))  # stock | crypto
    exchange: Mapped[Optional[str]] = mapped_column(String(32))
    currency: Mapped[Optional[str]] = mapped_column(String(8))


class OpportunityScore(Base):
    """A scored opportunity for an asset, derived from one or more events."""

    __tablename__ = "opportunity_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    asset_class: Mapped[str] = mapped_column(String(16))
    direction: Mapped[str] = mapped_column(String(8))  # buy | sell | watch | avoid
    total_score: Mapped[float] = mapped_column(Float)
    factors_json: Mapped[str] = mapped_column(Text)
    reasoning: Mapped[str] = mapped_column(Text)
    sources_json: Mapped[Optional[str]] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    risk_level: Mapped[str] = mapped_column(String(16), default="medium")


class PaperTrade(Base):
    """A simulated trade in paper trading mode."""

    __tablename__ = "paper_trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    opened_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    closed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    asset_class: Mapped[str] = mapped_column(String(16))
    side: Mapped[str] = mapped_column(String(8))  # buy/sell
    quantity: Mapped[float] = mapped_column(Float)
    entry_price: Mapped[float] = mapped_column(Float)
    stop_loss: Mapped[Optional[float]] = mapped_column(Float)
    take_profit: Mapped[Optional[float]] = mapped_column(Float)
    trailing_stop_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    high_watermark: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    exit_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    exit_reason: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    pnl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="open", index=True)  # open|closed|pending_approval|cancelled
    opportunity_id: Mapped[Optional[int]] = mapped_column(ForeignKey("opportunity_scores.id"))
    reasoning: Mapped[Optional[str]] = mapped_column(Text)
    approved_by: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)


class PortfolioSnapshot(Base):
    """Point-in-time portfolio value snapshot."""

    __tablename__ = "portfolio_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    cash: Mapped[float] = mapped_column(Float)
    positions_value: Mapped[float] = mapped_column(Float)
    total_value: Mapped[float] = mapped_column(Float)
    daily_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    open_positions: Mapped[int] = mapped_column(Integer, default=0)


class DecisionLog(Base):
    """Audit log of every decision (trade or no-trade) the bot makes."""

    __tablename__ = "decision_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    symbol: Mapped[Optional[str]] = mapped_column(String(32))
    action: Mapped[str] = mapped_column(String(32))  # buy | sell | skip | watch | recommend
    decision: Mapped[str] = mapped_column(String(32))  # approved | rejected | pending | simulated
    score: Mapped[Optional[float]] = mapped_column(Float)
    reason: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[Optional[str]] = mapped_column(Text)


class HighImpactAccount(Base):
    """Persisted high-impact account registry (synced from config)."""

    __tablename__ = "high_impact_accounts"
    __table_args__ = (UniqueConstraint("username", name="uq_high_impact_username"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(64), index=True)
    display_name: Mapped[Optional[str]] = mapped_column(String(128))
    category: Mapped[str] = mapped_column(String(32))
    credibility: Mapped[float] = mapped_column(Float, default=0.5)
    official: Mapped[bool] = mapped_column(Boolean, default=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True)


class Source(Base):
    """News/feed source credibility registry."""

    __tablename__ = "sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    domain: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    credibility: Mapped[float] = mapped_column(Float, default=0.5)
    notes: Mapped[Optional[str]] = mapped_column(Text)
