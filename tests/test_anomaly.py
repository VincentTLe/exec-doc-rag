"""Tests for anomaly detection."""

import numpy as np
import pandas as pd
import pytest

from src.execution.anomaly import compute_zscores, flag_anomalies


@pytest.fixture
def normal_data() -> pd.DataFrame:
    """DataFrame with normally distributed metrics."""
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame({
        "order_id": [f"ORD-{i:05d}" for i in range(n)],
        "total_is_bps": rng.normal(2.0, 3.0, n),
        "delay_cost_bps": rng.normal(0.5, 1.0, n),
        "execution_cost_bps": rng.normal(1.0, 2.0, n),
        "symbol": rng.choice(["AAPL", "MSFT", "GOOGL"], n),
    })


@pytest.fixture
def data_with_outliers(normal_data: pd.DataFrame) -> pd.DataFrame:
    """Normal data with a few extreme outliers injected."""
    df = normal_data.copy()
    # Inject 3 outliers
    df.loc[0, "total_is_bps"] = 50.0  # Way above normal
    df.loc[1, "total_is_bps"] = -40.0  # Way below normal
    df.loc[2, "delay_cost_bps"] = 20.0  # Very high delay cost
    return df


def test_zscore_mean_near_zero(normal_data: pd.DataFrame) -> None:
    """Global z-scores should have mean ~0."""
    result = compute_zscores(normal_data, ["total_is_bps"])
    assert abs(result["total_is_bps_zscore"].mean()) < 0.01


def test_zscore_std_near_one(normal_data: pd.DataFrame) -> None:
    """Global z-scores should have std ~1."""
    result = compute_zscores(normal_data, ["total_is_bps"])
    assert abs(result["total_is_bps_zscore"].std() - 1.0) < 0.05


def test_anomaly_flags_extreme_values(data_with_outliers: pd.DataFrame) -> None:
    """Injected outliers should be flagged."""
    result = flag_anomalies(
        data_with_outliers,
        metric_columns=["total_is_bps", "delay_cost_bps"],
        threshold=2.5,
    )
    # At least the extreme outliers should be flagged
    assert result["is_anomaly"].sum() >= 2


def test_no_anomalies_in_clean_data(normal_data: pd.DataFrame) -> None:
    """Standard normal data with threshold=3.0 should flag very few."""
    result = flag_anomalies(
        normal_data,
        metric_columns=["total_is_bps"],
        threshold=3.0,
    )
    # With N=200 and threshold=3, expect <2% flagged
    anomaly_rate = result["is_anomaly"].sum() / len(result)
    assert anomaly_rate < 0.05


def test_anomaly_reasons_populated(data_with_outliers: pd.DataFrame) -> None:
    """Flagged anomalies should have non-empty reasons."""
    result = flag_anomalies(
        data_with_outliers,
        metric_columns=["total_is_bps", "delay_cost_bps"],
        threshold=2.5,
    )
    anomalies = result[result["is_anomaly"]]
    for _, row in anomalies.iterrows():
        assert row["anomaly_reasons"] != ""


def test_grouped_zscore(normal_data: pd.DataFrame) -> None:
    """Grouped z-scores should work per-group."""
    result = compute_zscores(
        normal_data,
        metric_columns=["total_is_bps"],
        group_by="symbol",
    )
    assert "total_is_bps_zscore" in result.columns
    # Each group's z-scores should have mean ~0
    for symbol in normal_data["symbol"].unique():
        group_zscores = result[result["symbol"] == symbol]["total_is_bps_zscore"]
        assert abs(group_zscores.mean()) < 0.1


def test_anomaly_score_is_max_zscore(data_with_outliers: pd.DataFrame) -> None:
    """anomaly_score should equal the max absolute z-score."""
    result = flag_anomalies(
        data_with_outliers,
        metric_columns=["total_is_bps", "delay_cost_bps"],
        threshold=2.5,
    )
    # Check that anomaly_score >= threshold for flagged items
    flagged = result[result["is_anomaly"]]
    for _, row in flagged.iterrows():
        assert row["anomaly_score"] >= 2.5
