"""Generate synthetic trade execution data.

Creates realistic trade orders and fills with controllable anomaly injection.
The data mimics what a multi-manager hedge fund's execution desk would see:
diverse order types, multiple venues, realistic slippage distributions,
and occasional anomalous executions.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.execution.schemas import (
    BASE_PRICES,
    ORDER_TYPES,
    SIDES,
    SYMBOLS,
    URGENCY_LEVELS,
    VENUES,
)


def generate_dataset(
    n_orders: int = 1500,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate complete synthetic dataset.

    Returns:
        (orders_df, fills_df): DataFrames ready for analysis.
    """
    rng = np.random.default_rng(seed)

    orders_records: list[dict] = []
    fills_records: list[dict] = []

    # Generate over 20 trading days
    base_date = datetime(2024, 10, 1, 9, 30, 0)
    trading_days = [
        base_date + timedelta(days=d)
        for d in range(28)
        if (base_date + timedelta(days=d)).weekday() < 5  # Skip weekends
    ][:20]

    orders_per_day = n_orders // len(trading_days)
    anomaly_rate = 0.08  # 8% anomalous

    for day in trading_days:
        for i in range(orders_per_day):
            order_idx = len(orders_records)
            is_anomalous = rng.random() < anomaly_rate

            # Pick symbol and side
            symbol = rng.choice(SYMBOLS)
            side = rng.choice(SIDES, p=[0.52, 0.48])  # Slight buy bias
            base_price = BASE_PRICES[symbol]

            # Order type distribution: 55% MARKET, 25% LIMIT, 10% VWAP, 5% TWAP, 5% MOC
            order_type = rng.choice(
                ORDER_TYPES, p=[0.55, 0.25, 0.10, 0.05, 0.05]
            )

            # Order size: log-normal, centered around 500 shares
            quantity = int(np.clip(rng.lognormal(mean=6.2, sigma=0.8), 100, 50000))
            quantity = (quantity // 100) * 100  # Round to 100s
            quantity = max(quantity, 100)

            urgency = rng.choice(URGENCY_LEVELS, p=[0.3, 0.5, 0.2])

            # Decision price: base + daily drift + noise
            daily_drift = rng.normal(0, 0.005) * base_price
            decision_price = round(base_price + daily_drift + rng.normal(0, 0.1), 2)

            # Decision-to-arrival delay: 0-30 minutes, skewed short
            delay_minutes = max(0, rng.exponential(scale=5.0))
            delay_minutes = min(delay_minutes, 30)
            decision_time = day + timedelta(
                minutes=rng.uniform(0, 300)  # Random time in trading day
            )
            arrival_time = decision_time + timedelta(minutes=delay_minutes)

            # Arrival price: decision + drift during delay
            side_sign = 1 if side == "BUY" else -1
            delay_drift = side_sign * abs(rng.normal(0, 0.0002)) * decision_price * delay_minutes
            arrival_price = round(decision_price + delay_drift, 2)

            # Limit price for LIMIT orders
            limit_price = None
            if order_type == "LIMIT":
                if side == "BUY":
                    limit_price = round(decision_price * (1 + rng.uniform(0, 0.005)), 2)
                else:
                    limit_price = round(decision_price * (1 - rng.uniform(0, 0.005)), 2)

            order_id = f"ORD-{day.strftime('%Y%m%d')}-{i:05d}"

            # Generate fills for this order
            order_fills, close_price = _generate_fills(
                order_id=order_id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                arrival_price=arrival_price,
                arrival_time=arrival_time,
                is_anomalous=is_anomalous,
                rng=rng,
            )

            filled_qty = sum(f["fill_quantity"] for f in order_fills)
            total_commission = sum(f["commission"] for f in order_fills)
            vwap_fill = (
                sum(f["fill_price"] * f["fill_quantity"] for f in order_fills) / filled_qty
                if filled_qty > 0
                else 0.0
            )
            primary_venue = max(
                set(f["venue"] for f in order_fills),
                key=lambda v: sum(
                    f["fill_quantity"] for f in order_fills if f["venue"] == v
                ),
            ) if order_fills else ""

            orders_records.append({
                "order_id": order_id,
                "symbol": symbol,
                "side": side,
                "order_type": order_type,
                "quantity": quantity,
                "decision_price": round(decision_price, 4),
                "arrival_price": round(arrival_price, 4),
                "limit_price": limit_price,
                "vwap_fill_price": round(vwap_fill, 4),
                "filled_quantity": filled_qty,
                "unfilled_quantity": quantity - filled_qty,
                "close_price": round(close_price, 4),
                "total_commission": round(total_commission, 4),
                "num_fills": len(order_fills),
                "primary_venue": primary_venue,
                "decision_time": decision_time.isoformat(),
                "arrival_time": arrival_time.isoformat(),
                "urgency": urgency,
                "is_anomalous": is_anomalous,
            })

            fills_records.extend(order_fills)

    orders_df = pd.DataFrame(orders_records)
    fills_df = pd.DataFrame(fills_records)

    return orders_df, fills_df


def _generate_fills(
    order_id: str,
    symbol: str,
    side: str,
    order_type: str,
    quantity: int,
    arrival_price: float,
    arrival_time: datetime,
    is_anomalous: bool,
    rng: np.random.Generator,
) -> tuple[list[dict], float]:
    """Generate fills for a single order. Returns (fills, close_price).

    Characteristics:
    - Fill rate: 85-100% for market, 60-95% for limit
    - Partial fills: 1-8 child fills
    - Slippage: mean 0.5bps normal, 5-15bps for anomalies
    """
    fills: list[dict] = []

    # Determine fill rate
    if order_type == "MARKET":
        fill_rate = rng.uniform(0.95, 1.0)
    elif order_type == "LIMIT":
        fill_rate = rng.uniform(0.60, 0.95)
    else:  # VWAP, TWAP, MOC
        fill_rate = rng.uniform(0.85, 1.0)

    total_to_fill = int(quantity * fill_rate)
    total_to_fill = (total_to_fill // 100) * 100  # Round to 100s
    total_to_fill = max(total_to_fill, 100) if total_to_fill > 0 else 0

    if total_to_fill == 0:
        close_price = arrival_price * (1 + rng.normal(0, 0.003))
        return fills, round(close_price, 2)

    # Number of child fills
    n_fills = min(rng.integers(1, 9), max(1, total_to_fill // 100))

    # Split quantity across fills
    fill_sizes = _split_quantity(total_to_fill, n_fills, rng)

    # Generate each fill
    side_sign = 1 if side == "BUY" else -1
    current_price = arrival_price

    # Venue weights
    venue_weights = [0.30, 0.25, 0.15, 0.15, 0.05, 0.10]

    for j, fill_qty in enumerate(fill_sizes):
        # Time between fills: 1-30 minutes
        fill_delay = timedelta(minutes=rng.exponential(scale=8.0))
        fill_time = arrival_time + fill_delay * (j + 1)

        # Slippage model
        if is_anomalous:
            # Anomalous: 5-15 bps adverse slippage
            slippage_bps = side_sign * rng.uniform(5, 15)
        else:
            # Normal: mean 0.5bps, std 2bps
            slippage_bps = side_sign * rng.normal(0.5, 2.0)

        fill_price = current_price * (1 + slippage_bps / 10000)

        # Small random walk for next fill
        current_price *= 1 + rng.normal(0, 0.0003)

        venue = rng.choice(VENUES, p=venue_weights)

        # Commission: $0.003-0.005 per share
        commission_per_share = rng.uniform(0.003, 0.005)

        fills.append({
            "fill_id": f"FILL-{order_id}-{j:03d}",
            "order_id": order_id,
            "venue": venue,
            "fill_price": round(fill_price, 4),
            "fill_quantity": fill_qty,
            "fill_time": fill_time.isoformat(),
            "commission": round(commission_per_share * fill_qty, 4),
        })

    # Close price: arrival + random walk over remaining trading day
    hours_remaining = rng.uniform(1, 6)
    close_drift = rng.normal(0, 0.001 * np.sqrt(hours_remaining)) * arrival_price
    close_price = arrival_price + close_drift

    return fills, round(close_price, 2)


def _split_quantity(
    total: int,
    n_parts: int,
    rng: np.random.Generator,
) -> list[int]:
    """Split total quantity into n_parts, each a multiple of 100."""
    if n_parts <= 1:
        return [total]

    # Generate random proportions
    props = rng.dirichlet(np.ones(n_parts) * 2)
    raw_sizes = (props * total).astype(int)

    # Round each to nearest 100
    sizes = [(max(s, 100) // 100) * 100 for s in raw_sizes]

    # Adjust last fill to ensure total matches
    current_sum = sum(sizes[:-1])
    sizes[-1] = total - current_sum

    # Ensure last fill is positive and reasonable
    if sizes[-1] <= 0:
        sizes[-1] = 100
        # Redistribute from largest fills
        excess = sum(sizes) - total
        for i in range(len(sizes) - 1):
            if sizes[i] > 200 and excess > 0:
                reduction = min(100, excess)
                sizes[i] -= reduction
                excess -= reduction

    return [s for s in sizes if s > 0]
