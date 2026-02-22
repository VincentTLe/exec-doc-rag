"""Implementation Shortfall and execution quality metrics.

Implementation Shortfall (IS) decomposition following Perold (1988):

For a BUY order:
  Total IS = (Paper Return - Actual Return) / Paper Portfolio Value

Decomposed into:
  1. Delay Cost: Cost of waiting between decision and first fill
     = side_sign * (arrival - decision) * total_shares / paper_value
  2. Execution Cost (Market Impact): Cost of executing in market
     = side_sign * (vwap_fill - arrival) * filled_shares / paper_value
  3. Opportunity Cost: Cost of unfilled shares
     = side_sign * (close - decision) * unfilled_shares / paper_value
  4. Fixed Cost: Commissions and fees
     = total_commission / paper_value

All costs expressed in basis points (bps = 1/10000).
For SELL orders, the side sign is flipped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ISDecomposition:
    """Implementation Shortfall decomposition for a single order."""

    order_id: str
    total_is_bps: float
    delay_cost_bps: float
    execution_cost_bps: float
    opportunity_cost_bps: float
    fixed_cost_bps: float


def compute_is_single(
    order_id: str,
    side: str,
    decision_price: float,
    arrival_price: float,
    vwap_fill_price: float,
    close_price: float,
    total_shares: int,
    filled_shares: int,
    total_commission: float,
) -> ISDecomposition:
    """Compute IS decomposition for a single order.

    All results in basis points (bps).
    """
    if decision_price <= 0 or total_shares <= 0:
        return ISDecomposition(order_id, 0, 0, 0, 0, 0)

    side_sign = 1.0 if side == "BUY" else -1.0
    paper_value = decision_price * total_shares
    unfilled_shares = total_shares - filled_shares

    # Delay cost: price moved against us while we waited
    delay_cost = (
        side_sign
        * (arrival_price - decision_price)
        * total_shares
        / paper_value
        * 10000
    )

    # Execution cost (market impact): price moved against us during execution
    if filled_shares > 0:
        execution_cost = (
            side_sign
            * (vwap_fill_price - arrival_price)
            * filled_shares
            / paper_value
            * 10000
        )
    else:
        execution_cost = 0.0

    # Opportunity cost: missed the move on unfilled shares
    if unfilled_shares > 0:
        opportunity_cost = (
            side_sign
            * (close_price - decision_price)
            * unfilled_shares
            / paper_value
            * 10000
        )
    else:
        opportunity_cost = 0.0

    # Fixed costs: commissions and fees
    fixed_cost = total_commission / paper_value * 10000

    total_is = delay_cost + execution_cost + opportunity_cost + fixed_cost

    return ISDecomposition(
        order_id=order_id,
        total_is_bps=round(total_is, 4),
        delay_cost_bps=round(delay_cost, 4),
        execution_cost_bps=round(execution_cost, 4),
        opportunity_cost_bps=round(opportunity_cost, 4),
        fixed_cost_bps=round(fixed_cost, 4),
    )


def compute_is_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Compute IS decomposition for an entire orders DataFrame.

    Expects columns: order_id, side, decision_price, arrival_price,
    vwap_fill_price, close_price, quantity, filled_quantity, total_commission.

    Adds columns: total_is_bps, delay_cost_bps, execution_cost_bps,
    opportunity_cost_bps, fixed_cost_bps.
    """
    results: list[ISDecomposition] = []

    for _, row in df.iterrows():
        is_result = compute_is_single(
            order_id=row["order_id"],
            side=row["side"],
            decision_price=row["decision_price"],
            arrival_price=row["arrival_price"],
            vwap_fill_price=row.get("vwap_fill_price", row["arrival_price"]),
            close_price=row["close_price"],
            total_shares=row["quantity"],
            filled_shares=row.get("filled_quantity", row["quantity"]),
            total_commission=row.get("total_commission", 0),
        )
        results.append(is_result)

    is_df = pd.DataFrame([
        {
            "order_id": r.order_id,
            "total_is_bps": r.total_is_bps,
            "delay_cost_bps": r.delay_cost_bps,
            "execution_cost_bps": r.execution_cost_bps,
            "opportunity_cost_bps": r.opportunity_cost_bps,
            "fixed_cost_bps": r.fixed_cost_bps,
        }
        for r in results
    ])

    # Merge back into original DataFrame
    result_df = df.copy()
    for col in ["total_is_bps", "delay_cost_bps", "execution_cost_bps",
                 "opportunity_cost_bps", "fixed_cost_bps"]:
        result_df[col] = is_df[col].values

    return result_df


def compute_execution_summary(df: pd.DataFrame) -> dict:
    """Compute aggregate execution quality metrics.

    Expects a DataFrame with IS columns already computed.

    Returns dict with summary statistics.
    """
    summary = {
        "total_orders": len(df),
        "avg_is_bps": round(df["total_is_bps"].mean(), 2),
        "median_is_bps": round(df["total_is_bps"].median(), 2),
        "std_is_bps": round(df["total_is_bps"].std(), 2),
        "p5_is_bps": round(df["total_is_bps"].quantile(0.05), 2),
        "p95_is_bps": round(df["total_is_bps"].quantile(0.95), 2),
        "avg_delay_cost": round(df["delay_cost_bps"].mean(), 2),
        "avg_execution_cost": round(df["execution_cost_bps"].mean(), 2),
        "avg_opportunity_cost": round(df["opportunity_cost_bps"].mean(), 2),
        "avg_fixed_cost": round(df["fixed_cost_bps"].mean(), 2),
    }

    # Fill rate
    if "filled_quantity" in df.columns and "quantity" in df.columns:
        fill_rates = df["filled_quantity"] / df["quantity"]
        summary["avg_fill_rate"] = round(fill_rates.mean(), 4)
        summary["min_fill_rate"] = round(fill_rates.min(), 4)

    # Breakdown by order type
    if "order_type" in df.columns:
        summary["by_order_type"] = (
            df.groupby("order_type")["total_is_bps"]
            .agg(["mean", "median", "count"])
            .round(2)
            .to_dict()
        )

    # Breakdown by venue
    if "primary_venue" in df.columns:
        summary["by_venue"] = (
            df.groupby("primary_venue")["total_is_bps"]
            .agg(["mean", "median", "count"])
            .round(2)
            .to_dict()
        )

    return summary
