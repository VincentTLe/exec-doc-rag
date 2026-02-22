"""Z-score anomaly detection on execution quality metrics.

Flags orders where execution metrics deviate significantly from
the population distribution. Supports both global and grouped
(per-symbol, per-order-type) z-score computation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_zscores(
    df: pd.DataFrame,
    metric_columns: list[str],
    group_by: str | None = None,
) -> pd.DataFrame:
    """Compute z-scores for specified metric columns.

    If group_by is provided, z-scores are computed within each group
    (e.g., per-symbol normalizes within each symbol's distribution).

    Adds columns: {metric}_zscore for each metric.
    """
    result = df.copy()

    for col in metric_columns:
        if col not in df.columns:
            continue

        zscore_col = f"{col}_zscore"

        if group_by and group_by in df.columns:
            # Grouped z-score
            grouped = df.groupby(group_by)[col]
            result[zscore_col] = grouped.transform(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
            )
        else:
            # Global z-score
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                result[zscore_col] = (df[col] - mean) / std
            else:
                result[zscore_col] = 0.0

    return result


def flag_anomalies(
    df: pd.DataFrame,
    metric_columns: list[str],
    threshold: float = 2.5,
    group_by: str | None = None,
) -> pd.DataFrame:
    """Flag orders where any metric z-score exceeds threshold.

    Adds columns:
    - is_anomaly: bool
    - anomaly_score: float (max absolute z-score across metrics)
    - anomaly_reasons: str (comma-separated list of flagged metrics)

    Args:
        df: DataFrame with execution metrics.
        metric_columns: Columns to check for anomalies.
        threshold: Z-score threshold for flagging.
        group_by: Optional column for grouped z-score computation.

    Returns:
        DataFrame with anomaly columns added.
    """
    # Compute z-scores
    result = compute_zscores(df, metric_columns, group_by)

    zscore_cols = [f"{col}_zscore" for col in metric_columns if f"{col}_zscore" in result.columns]

    # Flag based on threshold
    anomaly_flags = result[zscore_cols].abs() > threshold

    result["is_anomaly"] = anomaly_flags.any(axis=1)
    result["anomaly_score"] = result[zscore_cols].abs().max(axis=1)

    # Build reason strings
    reasons: list[str] = []
    for _, row in anomaly_flags.iterrows():
        flagged = [col.replace("_zscore", "") for col in zscore_cols if row.get(col, False)]
        reasons.append(", ".join(flagged) if flagged else "")
    result["anomaly_reasons"] = reasons

    return result


def anomaly_summary(df: pd.DataFrame) -> dict:
    """Summarize anomalies in a flagged DataFrame.

    Expects columns: is_anomaly, anomaly_score, anomaly_reasons.

    Returns summary dict with counts, distributions, and top offenders.
    """
    if "is_anomaly" not in df.columns:
        return {"error": "DataFrame not flagged — run flag_anomalies first"}

    anomalies = df[df["is_anomaly"]]

    summary = {
        "total_orders": len(df),
        "total_anomalies": len(anomalies),
        "anomaly_rate": round(len(anomalies) / len(df), 4) if len(df) > 0 else 0,
        "avg_anomaly_score": round(anomalies["anomaly_score"].mean(), 2) if len(anomalies) > 0 else 0,
    }

    # Most common reasons
    if len(anomalies) > 0 and "anomaly_reasons" in anomalies.columns:
        all_reasons: list[str] = []
        for reasons_str in anomalies["anomaly_reasons"]:
            if reasons_str:
                all_reasons.extend(reasons_str.split(", "))
        if all_reasons:
            reason_counts = pd.Series(all_reasons).value_counts()
            summary["top_reasons"] = reason_counts.head(5).to_dict()

    # Distribution by symbol
    if "symbol" in df.columns and len(anomalies) > 0:
        summary["by_symbol"] = anomalies["symbol"].value_counts().head(10).to_dict()

    # Distribution by order type
    if "order_type" in df.columns and len(anomalies) > 0:
        summary["by_order_type"] = anomalies["order_type"].value_counts().to_dict()

    # Top offenders
    if len(anomalies) > 0 and "total_is_bps" in anomalies.columns:
        worst = anomalies.nlargest(10, "anomaly_score")
        summary["worst_10"] = worst[
            ["order_id", "symbol", "total_is_bps", "anomaly_score", "anomaly_reasons"]
        ].to_dict("records")

    return summary
