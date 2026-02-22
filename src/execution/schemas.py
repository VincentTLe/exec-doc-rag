"""Data classes for trade execution analytics.

Defines the core data structures for orders, fills, and execution summaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


# Constant pools for data generation and validation
SYMBOLS = (
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA",
    "JPM", "BAC", "GS", "MS", "V", "MA",
    "JNJ", "PFE", "XOM", "CVX", "SPY", "QQQ",
)

ORDER_TYPES = ("MARKET", "LIMIT", "VWAP", "TWAP", "MOC")

VENUES = ("NYSE", "NASDAQ", "ARCA", "BATS", "IEX", "DARK_POOL")

SIDES = ("BUY", "SELL")

URGENCY_LEVELS = ("LOW", "MEDIUM", "HIGH")

# Approximate base prices for realistic data generation
BASE_PRICES: dict[str, float] = {
    "AAPL": 185.0, "MSFT": 420.0, "GOOGL": 175.0, "AMZN": 190.0,
    "META": 520.0, "NVDA": 880.0, "JPM": 195.0, "BAC": 38.0,
    "GS": 450.0, "MS": 95.0, "V": 280.0, "MA": 460.0,
    "JNJ": 155.0, "PFE": 27.0, "XOM": 105.0, "CVX": 155.0,
    "SPY": 510.0, "QQQ": 440.0,
}


@dataclass
class Order:
    """A single trade order."""

    order_id: str
    symbol: str
    side: str  # BUY or SELL
    order_type: str  # MARKET, LIMIT, VWAP, TWAP, MOC
    quantity: int  # Total shares ordered
    decision_price: float  # Price at decision time
    arrival_price: float  # Price when order hits market
    limit_price: float | None  # For LIMIT orders only
    decision_time: datetime
    arrival_time: datetime
    urgency: str  # LOW, MEDIUM, HIGH
    is_anomalous: bool  # Whether this is an injected anomaly


@dataclass
class Fill:
    """A single fill (partial execution) of an order."""

    fill_id: str
    order_id: str
    venue: str
    fill_price: float
    fill_quantity: int
    fill_time: datetime
    commission: float  # USD
