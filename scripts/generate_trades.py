"""Generate synthetic trade execution data."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import TRADES_DIR
from src.execution.data_generator import generate_dataset

if __name__ == "__main__":
    TRADES_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic trade data...")
    orders_df, fills_df = generate_dataset(n_orders=1500, seed=42)

    orders_path = TRADES_DIR / "orders.csv"
    fills_path = TRADES_DIR / "fills.csv"

    orders_df.to_csv(orders_path, index=False)
    fills_df.to_csv(fills_path, index=False)

    print(f"Generated {len(orders_df)} orders -> {orders_path}")
    print(f"Generated {len(fills_df)} fills -> {fills_path}")

    # Print summary statistics
    print(f"\nOrder types: {orders_df['order_type'].value_counts().to_dict()}")
    print(f"Sides: {orders_df['side'].value_counts().to_dict()}")
    print(f"Symbols: {orders_df['symbol'].nunique()} unique")
    print(f"Anomalous orders: {orders_df['is_anomalous'].sum()}")
