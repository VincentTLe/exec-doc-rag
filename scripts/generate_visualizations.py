"""Generate visualization charts for ExecDocRAG reports.

Creates publication-quality charts:
1. Retrieval evaluation metrics (Recall@k, MRR, NDCG@k)
2. Evaluation breakdown by difficulty
3. Implementation Shortfall decomposition
4. IS distribution with anomaly detection
5. Venue performance comparison
6. Daily execution quality trends

Outputs PNG files to reports/ directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_DIR = PROJECT_ROOT / "data"

# Style configuration
COLORS = {
    "primary": "#2563EB",     # Blue
    "secondary": "#7C3AED",   # Purple
    "success": "#059669",     # Green
    "warning": "#D97706",     # Amber
    "danger": "#DC2626",      # Red
    "neutral": "#6B7280",     # Gray
    "light": "#F3F4F6",       # Light gray
}

COMPONENT_COLORS = {
    "Delay Cost": "#3B82F6",
    "Execution Cost": "#8B5CF6",
    "Opportunity Cost": "#F59E0B",
    "Fixed Cost": "#6B7280",
}

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
})


def plot_retrieval_metrics(output_path: Path) -> None:
    """Bar chart of retrieval evaluation metrics."""
    metrics = {
        "Recall@3": 0.633,
        "Recall@5": 0.667,
        "Recall@10": 0.700,
        "MRR": 0.516,
        "NDCG@3": 0.450,
        "NDCG@5": 0.465,
        "NDCG@10": 0.554,
    }

    fig, ax = plt.subplots(figsize=(10, 5))

    names = list(metrics.keys())
    values = list(metrics.values())
    colors = [COLORS["primary"]] * 3 + [COLORS["secondary"]] + [COLORS["success"]] * 3

    bars = ax.bar(names, values, color=colors, edgecolor="white", linewidth=0.5, width=0.65)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{val:.1%}" if val < 1 else f"{val:.3f}",
            ha="center", va="bottom", fontweight="bold", fontsize=11,
        )

    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Retrieval Evaluation Metrics (30 Questions)", fontsize=14, fontweight="bold", pad=15)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS["primary"], label="Recall@k"),
        mpatches.Patch(facecolor=COLORS["secondary"], label="MRR"),
        mpatches.Patch(facecolor=COLORS["success"], label="NDCG@k"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_difficulty_breakdown(output_path: Path) -> None:
    """Grouped bar chart of metrics by difficulty level."""
    data = {
        "Easy (n=10)": {"Recall@3": 0.700, "Recall@5": 0.700, "MRR": 0.583},
        "Medium (n=12)": {"Recall@3": 0.667, "Recall@5": 0.750, "MRR": 0.597},
        "Hard (n=8)": {"Recall@3": 0.500, "Recall@5": 0.500, "MRR": 0.312},
    }

    fig, ax = plt.subplots(figsize=(9, 5))

    difficulties = list(data.keys())
    metric_names = ["Recall@3", "Recall@5", "MRR"]
    x = np.arange(len(difficulties))
    width = 0.25
    metric_colors = [COLORS["primary"], COLORS["success"], COLORS["secondary"]]

    for i, (metric, color) in enumerate(zip(metric_names, metric_colors)):
        values = [data[d][metric] for d in difficulties]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width, label=metric, color=color, edgecolor="white")
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.0%}",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Retrieval Performance by Question Difficulty", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(difficulties, fontsize=11)
    ax.set_ylim(0, 0.95)
    ax.legend(framealpha=0.9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_is_decomposition(orders_df: pd.DataFrame, output_path: Path) -> None:
    """Stacked bar chart of IS component averages by venue."""
    components = ["delay_cost_bps", "execution_cost_bps", "opportunity_cost_bps", "fixed_cost_bps"]
    component_labels = ["Delay Cost", "Execution Cost", "Opportunity Cost", "Fixed Cost"]

    # Filter to orders that have fills (non-zero IS)
    filled = orders_df[orders_df["filled_quantity"] > 0].copy()

    if "primary_venue" not in filled.columns or filled.empty:
        print("  [SKIP] No venue data for IS decomposition chart")
        return

    venue_means = filled.groupby("primary_venue")[components].mean()
    venue_means = venue_means.sort_values("delay_cost_bps", ascending=False)

    # Limit to top venues by order count
    venue_counts = filled["primary_venue"].value_counts()
    top_venues = venue_counts.head(8).index
    venue_means = venue_means.loc[venue_means.index.isin(top_venues)]

    fig, ax = plt.subplots(figsize=(11, 6))

    bottoms = np.zeros(len(venue_means))
    for comp, label in zip(components, component_labels):
        vals = venue_means[comp].values
        ax.bar(
            venue_means.index, vals, bottom=bottoms,
            label=label, color=COMPONENT_COLORS[label],
            edgecolor="white", linewidth=0.5, width=0.6,
        )
        bottoms += vals

    # Add total labels
    for i, venue in enumerate(venue_means.index):
        total = venue_means.loc[venue, components].sum()
        ax.text(i, total + 0.3, f"{total:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Average Cost (bps)", fontsize=12)
    ax.set_title("Implementation Shortfall Decomposition by Venue", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xlabel("Execution Venue", fontsize=12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=0)
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_is_distribution(orders_df: pd.DataFrame, output_path: Path) -> None:
    """Histogram of IS distribution with anomaly overlay."""
    filled = orders_df[orders_df["filled_quantity"] > 0].copy()
    if filled.empty:
        print("  [SKIP] No filled orders for IS distribution")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    is_values = filled["total_is_bps"]

    # Clip extreme outliers for visualization
    p1, p99 = is_values.quantile(0.01), is_values.quantile(0.99)
    clip_vals = is_values.clip(p1, p99)

    # Plot histogram for normal orders
    normal = filled[~filled["is_anomalous"]]
    anomalous = filled[filled["is_anomalous"]]

    bins = np.linspace(clip_vals.min(), clip_vals.max(), 50)

    ax.hist(
        normal["total_is_bps"].clip(p1, p99), bins=bins,
        color=COLORS["primary"], alpha=0.7, label=f"Normal (n={len(normal)})",
        edgecolor="white", linewidth=0.3,
    )
    if len(anomalous) > 0:
        ax.hist(
            anomalous["total_is_bps"].clip(p1, p99), bins=bins,
            color=COLORS["danger"], alpha=0.8, label=f"Anomalous (n={len(anomalous)})",
            edgecolor="white", linewidth=0.3,
        )

    # Statistics
    mean_is = is_values.mean()
    std_is = is_values.std()
    ax.axvline(mean_is, color=COLORS["warning"], linestyle="--", linewidth=2, label=f"Mean: {mean_is:.1f} bps")

    ax.set_xlabel("Total Implementation Shortfall (bps)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("IS Distribution with Anomaly Detection", fontsize=14, fontweight="bold", pad=15)
    ax.legend(framealpha=0.9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_daily_trends(orders_df: pd.DataFrame, output_path: Path) -> None:
    """Line chart of daily execution quality trends."""
    df = orders_df.copy()
    df["trade_date"] = pd.to_datetime(df["decision_time"]).dt.date

    daily = df.groupby("trade_date").agg(
        order_count=("order_id", "count"),
        avg_is_bps=("total_is_bps", "mean"),
        anomaly_count=("is_anomalous", "sum"),
    ).reset_index()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                     gridspec_kw={"height_ratios": [2, 1]})

    # Top: Avg IS with rolling average
    ax1.plot(daily["trade_date"], daily["avg_is_bps"], color=COLORS["primary"],
             linewidth=1.5, alpha=0.6, marker="o", markersize=4, label="Daily Avg IS")

    if len(daily) >= 3:
        rolling = daily["avg_is_bps"].rolling(3, min_periods=1).mean()
        ax1.plot(daily["trade_date"], rolling, color=COLORS["danger"],
                 linewidth=2.5, label="3-Day Moving Avg")

    ax1.set_ylabel("Avg IS (bps)", fontsize=12)
    ax1.set_title("Daily Execution Quality Trends", fontsize=14, fontweight="bold", pad=15)
    ax1.legend(framealpha=0.9)
    ax1.axhline(0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Bottom: Order count and anomaly count
    ax2.bar(daily["trade_date"], daily["order_count"], color=COLORS["primary"],
            alpha=0.5, label="Total Orders", width=0.8)
    ax2.bar(daily["trade_date"], daily["anomaly_count"], color=COLORS["danger"],
            alpha=0.8, label="Anomalies", width=0.8)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_xlabel("Trade Date", fontsize=12)
    ax2.legend(framealpha=0.9, loc="upper right")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_venue_performance(orders_df: pd.DataFrame, fills_df: pd.DataFrame, output_path: Path) -> None:
    """Bubble chart of venue performance: fill count vs avg IS, bubble size = volume."""
    filled = orders_df[orders_df["filled_quantity"] > 0].copy()
    if filled.empty or "primary_venue" not in filled.columns:
        print("  [SKIP] No venue data")
        return

    venue_stats = filled.groupby("primary_venue").agg(
        order_count=("order_id", "count"),
        avg_is_bps=("total_is_bps", "mean"),
        total_volume=("filled_quantity", "sum"),
        anomaly_rate=("is_anomalous", "mean"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Normalize bubble sizes
    max_vol = venue_stats["total_volume"].max()
    sizes = (venue_stats["total_volume"] / max_vol) * 1500 + 100

    # Color by anomaly rate
    scatter = ax.scatter(
        venue_stats["order_count"],
        venue_stats["avg_is_bps"],
        s=sizes,
        c=venue_stats["anomaly_rate"],
        cmap="RdYlGn_r",
        alpha=0.75,
        edgecolors="white",
        linewidth=1.5,
        vmin=0, vmax=0.2,
    )

    # Label each venue
    for _, row in venue_stats.iterrows():
        ax.annotate(
            row["primary_venue"],
            (row["order_count"], row["avg_is_bps"]),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center", fontsize=10, fontweight="bold",
        )

    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Anomaly Rate", fontsize=11)

    ax.set_xlabel("Number of Orders", fontsize=12)
    ax.set_ylabel("Average IS (bps)", fontsize=12)
    ax.set_title("Venue Performance: Orders vs IS (bubble = volume)", fontsize=14, fontweight="bold", pad=15)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_is_component_box(orders_df: pd.DataFrame, output_path: Path) -> None:
    """Box plot of IS components distribution."""
    filled = orders_df[orders_df["filled_quantity"] > 0].copy()
    if filled.empty:
        print("  [SKIP] No filled orders for box plot")
        return

    components = {
        "Delay": "delay_cost_bps",
        "Execution": "execution_cost_bps",
        "Opportunity": "opportunity_cost_bps",
        "Fixed": "fixed_cost_bps",
        "Total IS": "total_is_bps",
    }

    plot_data = []
    for label, col in components.items():
        for val in filled[col]:
            plot_data.append({"Component": label, "Cost (bps)": val})

    plot_df = pd.DataFrame(plot_data)

    fig, ax = plt.subplots(figsize=(10, 5))

    palette = ["#3B82F6", "#8B5CF6", "#F59E0B", "#6B7280", "#DC2626"]
    sns.boxplot(
        data=plot_df, x="Component", y="Cost (bps)",
        palette=palette, ax=ax, showfliers=False,
        width=0.6, linewidth=1.2,
    )

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title("IS Component Distributions (Filled Orders)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("")
    ax.set_ylabel("Cost (bps)", fontsize=12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def main() -> None:
    """Generate all visualization charts."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating ExecDocRAG Visualizations")
    print("=" * 45)

    # --- Retrieval Evaluation Charts ---
    print("\n[1/6] Retrieval metrics bar chart...")
    plot_retrieval_metrics(REPORTS_DIR / "retrieval_metrics.png")

    print("[2/6] Difficulty breakdown chart...")
    plot_difficulty_breakdown(REPORTS_DIR / "difficulty_breakdown.png")

    # --- Execution Analytics Charts ---
    orders_path = DATA_DIR / "synthetic_trades" / "orders.csv"
    fills_path = DATA_DIR / "synthetic_trades" / "fills.csv"

    if not orders_path.exists() or not fills_path.exists():
        print("\n[SKIP] Synthetic trade data not found. Run generate_trades.py first.")
        return

    print("\n  Loading trade data...")
    orders_df = pd.read_csv(orders_path)
    fills_df = pd.read_csv(fills_path)

    # Compute IS if not present
    if "total_is_bps" not in orders_df.columns:
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.execution.metrics import compute_is_batch
        orders_df = compute_is_batch(orders_df)

    print(f"  Loaded {len(orders_df)} orders, {len(fills_df)} fills\n")

    print("[3/6] IS decomposition by venue...")
    plot_is_decomposition(orders_df, REPORTS_DIR / "is_decomposition.png")

    print("[4/6] IS distribution with anomalies...")
    plot_is_distribution(orders_df, REPORTS_DIR / "is_distribution.png")

    print("[5/6] Daily execution trends...")
    plot_daily_trends(orders_df, REPORTS_DIR / "daily_trends.png")

    print("[6/6] Venue performance bubble chart...")
    plot_venue_performance(orders_df, fills_df, REPORTS_DIR / "venue_performance.png")

    print("\n" + "=" * 45)
    print(f"All charts saved to {REPORTS_DIR}/")
    print("Done!")


if __name__ == "__main__":
    main()
