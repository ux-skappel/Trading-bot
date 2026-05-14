"""Configuration for the trading bot.

Loads settings from environment variables (.env) with safe defaults.
Safety-critical flags default to the most conservative value.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except ImportError:  # pragma: no cover — optional dep
    def load_dotenv(*_args, **_kwargs) -> bool:  # type: ignore
        return False


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _str(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def _csv(name: str, default: List[str]) -> List[str]:
    raw = os.getenv(name)
    if not raw:
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass
class Settings:
    """Runtime configuration for the trading bot."""

    # --- Safety ---
    live_trading: bool = field(default_factory=lambda: _bool("LIVE_TRADING", False))
    require_manual_confirmation: bool = field(
        default_factory=lambda: _bool("REQUIRE_MANUAL_CONFIRMATION", True)
    )

    # --- Mode ---
    # analysis | paper | live
    trading_mode: str = field(default_factory=lambda: _str("TRADING_MODE", "paper"))
    human_approval_mode: bool = field(
        default_factory=lambda: _bool("HUMAN_APPROVAL_MODE", True)
    )

    # --- API keys ---
    x_bearer_token: str = field(default_factory=lambda: _str("X_BEARER_TOKEN"))
    reddit_client_id: str = field(default_factory=lambda: _str("REDDIT_CLIENT_ID"))
    reddit_client_secret: str = field(default_factory=lambda: _str("REDDIT_CLIENT_SECRET"))
    reddit_user_agent: str = field(
        default_factory=lambda: _str("REDDIT_USER_AGENT", "trading-bot-mvp/0.1")
    )
    reddit_subreddits: List[str] = field(
        default_factory=lambda: _csv(
            "REDDIT_SUBREDDITS",
            ["stocks", "wallstreetbets", "CryptoCurrency", "investing"],
        )
    )

    market_data_provider: str = field(
        default_factory=lambda: _str("MARKET_DATA_PROVIDER", "yfinance")
    )
    market_data_api_key: str = field(default_factory=lambda: _str("MARKET_DATA_API_KEY"))

    # --- Broker ---
    broker: str = field(default_factory=lambda: _str("BROKER", "paper"))
    alpaca_api_key: str = field(default_factory=lambda: _str("ALPACA_API_KEY"))
    alpaca_api_secret: str = field(default_factory=lambda: _str("ALPACA_API_SECRET"))
    alpaca_base_url: str = field(
        default_factory=lambda: _str("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    )

    # --- Storage ---
    database_url: str = field(
        default_factory=lambda: _str("DATABASE_URL", "sqlite:///./trading_bot.db")
    )

    log_level: str = field(default_factory=lambda: _str("LOG_LEVEL", "INFO"))

    # --- Portfolio / risk ---
    starting_cash: float = 100_000.0
    max_position_pct: float = 0.05          # 5% of portfolio per position
    max_daily_loss_pct: float = 0.03        # halt new trades if daily PnL <= -3%
    max_open_positions: int = 10
    max_exposure_per_class_pct: float = 0.4  # max 40% per asset class (stock/crypto)
    cooldown_minutes_after_loss: int = 60
    extreme_volatility_pct: float = 0.10    # block trades if daily ATR > 10% unless overridden
    allow_extreme_volatility: bool = False

    # --- Trading thresholds ---
    min_score_to_trade: float = 70.0
    min_score_to_recommend: float = 50.0
    min_credibility_to_trade: float = 0.7
    default_stop_loss_pct: float = 0.05
    default_take_profit_pct: float = 0.10
    use_trailing_stop: bool = True
    trailing_stop_pct: float = 0.04

    # --- Watchlists ---
    stock_watchlist: List[str] = field(
        default_factory=lambda: [
            # US large caps
            "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META", "AMD",
            "COIN", "MSTR", "PLTR", "JPM", "XOM",
            # Norwegian (Oslo, .OL suffix on yfinance)
            "EQNR.OL", "DNB.OL", "TEL.OL", "MOWI.OL", "AKERBP.OL", "YAR.OL",
            # European
            "ASML", "SAP", "NESN.SW", "MC.PA",
        ]
    )
    crypto_watchlist: List[str] = field(
        default_factory=lambda: ["BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "AVAX", "LINK"]
    )
    blacklist: List[str] = field(default_factory=list)

    # --- High-impact accounts ---
    # username -> (display_name, category, base_credibility 0..1, official)
    high_impact_accounts: Dict[str, Dict] = field(
        default_factory=lambda: {
            "realDonaldTrump":   {"name": "Donald Trump",        "category": "politician",   "credibility": 0.55, "official": False},
            "POTUS":             {"name": "US President",        "category": "government",   "credibility": 0.95, "official": True},
            "WhiteHouse":        {"name": "White House",         "category": "government",   "credibility": 0.95, "official": True},
            "USTreasury":        {"name": "US Treasury",         "category": "government",   "credibility": 0.95, "official": True},
            "federalreserve":    {"name": "Federal Reserve",     "category": "central_bank", "credibility": 0.98, "official": True},
            "ecb":               {"name": "European Central Bank","category": "central_bank","credibility": 0.98, "official": True},
            "bankofengland":     {"name": "Bank of England",     "category": "central_bank", "credibility": 0.98, "official": True},
            "SECGov":            {"name": "US SEC",              "category": "regulator",    "credibility": 0.97, "official": True},
            "CFTC":              {"name": "CFTC",                "category": "regulator",    "credibility": 0.95, "official": True},
            "FTC":               {"name": "US FTC",              "category": "regulator",    "credibility": 0.95, "official": True},
            "EU_Commission":     {"name": "EU Commission",       "category": "regulator",    "credibility": 0.95, "official": True},
            "elonmusk":          {"name": "Elon Musk",           "category": "ceo",          "credibility": 0.55, "official": False},
            "tim_cook":          {"name": "Tim Cook",            "category": "ceo",          "credibility": 0.80, "official": False},
            "jeffbezos":         {"name": "Jeff Bezos",          "category": "ceo",          "credibility": 0.70, "official": False},
            "Tesla":             {"name": "Tesla (official)",    "category": "company",      "credibility": 0.95, "official": True},
            "Apple":             {"name": "Apple (official)",    "category": "company",      "credibility": 0.95, "official": True},
            "saylor":            {"name": "Michael Saylor",      "category": "ceo",          "credibility": 0.55, "official": False},
            "VitalikButerin":    {"name": "Vitalik Buterin",     "category": "crypto_founder","credibility": 0.65, "official": False},
            "cz_binance":        {"name": "CZ (Binance)",        "category": "crypto_exec",  "credibility": 0.55, "official": False},
            "brian_armstrong":   {"name": "Brian Armstrong",     "category": "crypto_exec",  "credibility": 0.70, "official": False},
        }
    )

    # --- Source credibility (RSS / news domains) ---
    source_credibility: Dict[str, float] = field(
        default_factory=lambda: {
            "reuters.com":        0.95,
            "bloomberg.com":      0.92,
            "ft.com":             0.92,
            "wsj.com":            0.92,
            "apnews.com":         0.93,
            "cnbc.com":           0.80,
            "marketwatch.com":    0.75,
            "coindesk.com":       0.80,
            "cointelegraph.com":  0.65,
            "decrypt.co":         0.65,
            "yahoo.com":          0.65,
            "seekingalpha.com":   0.55,
            "benzinga.com":       0.55,
            "reddit.com":         0.30,
            "x.com":              0.30,
            "twitter.com":        0.30,
        }
    )

    # --- RSS feeds (free, public) ---
    rss_feeds: List[str] = field(
        default_factory=lambda: [
            "https://feeds.reuters.com/reuters/businessNews",
            "https://feeds.reuters.com/reuters/technologyNews",
            "https://feeds.bbci.co.uk/news/business/rss.xml",
            "https://www.cnbc.com/id/100003114/device/rss/rss.html",
            "https://www.marketwatch.com/rss/topstories",
            "https://www.coindesk.com/arc/outboundfeeds/rss/",
            "https://cointelegraph.com/rss",
            "https://decrypt.co/feed",
        ]
    )

    # --- Scoring weights (must sum to ~1.0) ---
    scoring_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "sentiment":            0.15,
            "source_credibility":   0.18,
            "recency":              0.08,
            "relevance":            0.12,
            "expected_impact":      0.18,
            "volatility":           0.05,
            "volume_change":        0.05,
            "price_momentum":       0.05,
            "duplicate_confirmation": 0.09,
            "risk_level":           0.05,
        }
    )

    def safety_summary(self) -> Dict[str, object]:
        """Return a dict of the safety-critical flags for display."""
        return {
            "live_trading": self.live_trading,
            "require_manual_confirmation": self.require_manual_confirmation,
            "trading_mode": self.trading_mode,
            "human_approval_mode": self.human_approval_mode,
            "broker": self.broker,
        }


SETTINGS = Settings()


def get_settings() -> Settings:
    """Return the singleton settings instance."""
    return SETTINGS
