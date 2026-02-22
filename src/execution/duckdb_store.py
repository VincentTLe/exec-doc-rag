"""DuckDB analytical store for trade execution data.

Provides fast SQL analytics over orders and fills without loading
everything into pandas. DuckDB excels at analytical queries and
demonstrates data engineering skills relevant to the role.
"""

from __future__ import annotations

import duckdb
import pandas as pd


class TradeStore:
    """DuckDB-backed analytical query engine for trade data."""

    def __init__(self, db_path: str = ":memory:"):
        """Initialize DuckDB connection (in-memory by default)."""
        self.conn = duckdb.connect(db_path)

    def load_data(
        self,
        orders_df: pd.DataFrame,
        fills_df: pd.DataFrame,
    ) -> None:
        """Load DataFrames into DuckDB tables."""
        self.conn.execute("DROP TABLE IF EXISTS orders")
        self.conn.execute("DROP TABLE IF EXISTS fills")
        self.conn.register("orders_view", orders_df)
        self.conn.register("fills_view", fills_df)
        self.conn.execute("CREATE TABLE orders AS SELECT * FROM orders_view")
        self.conn.execute("CREATE TABLE fills AS SELECT * FROM fills_view")

    def query(self, sql: str) -> pd.DataFrame:
        """Execute arbitrary SQL and return DataFrame."""
        return self.conn.execute(sql).fetchdf()

    def avg_is_by_venue(self) -> pd.DataFrame:
        """Average implementation shortfall by execution venue."""
        return self.query("""
            SELECT
                primary_venue AS venue,
                COUNT(*) AS order_count,
                ROUND(AVG(total_is_bps), 2) AS avg_is_bps,
                ROUND(MEDIAN(total_is_bps), 2) AS median_is_bps,
                ROUND(STDDEV(total_is_bps), 2) AS std_is_bps
            FROM orders
            GROUP BY primary_venue
            ORDER BY avg_is_bps DESC
        """)

    def avg_is_by_symbol(self) -> pd.DataFrame:
        """Average IS by symbol."""
        return self.query("""
            SELECT
                symbol,
                COUNT(*) AS order_count,
                ROUND(AVG(total_is_bps), 2) AS avg_is_bps,
                ROUND(MEDIAN(total_is_bps), 2) AS median_is_bps,
                ROUND(AVG(CAST(filled_quantity AS DOUBLE) / quantity), 4) AS avg_fill_rate
            FROM orders
            GROUP BY symbol
            ORDER BY avg_is_bps DESC
        """)

    def fill_rate_by_order_type(self) -> pd.DataFrame:
        """Fill rate breakdown by order type."""
        return self.query("""
            SELECT
                order_type,
                COUNT(*) AS order_count,
                ROUND(AVG(CAST(filled_quantity AS DOUBLE) / quantity), 4) AS avg_fill_rate,
                ROUND(AVG(total_is_bps), 2) AS avg_is_bps
            FROM orders
            GROUP BY order_type
            ORDER BY avg_fill_rate DESC
        """)

    def worst_executions(self, top_n: int = 20) -> pd.DataFrame:
        """Worst N executions by total IS."""
        return self.query(f"""
            SELECT
                order_id,
                symbol,
                side,
                order_type,
                quantity,
                ROUND(total_is_bps, 2) AS total_is_bps,
                ROUND(delay_cost_bps, 2) AS delay_cost,
                ROUND(execution_cost_bps, 2) AS exec_cost,
                ROUND(opportunity_cost_bps, 2) AS opp_cost,
                primary_venue,
                is_anomalous
            FROM orders
            ORDER BY total_is_bps DESC
            LIMIT {top_n}
        """)

    def daily_summary(self) -> pd.DataFrame:
        """Daily execution quality summary."""
        return self.query("""
            SELECT
                CAST(decision_time AS DATE) AS trade_date,
                COUNT(*) AS order_count,
                ROUND(AVG(total_is_bps), 2) AS avg_is_bps,
                ROUND(AVG(CAST(filled_quantity AS DOUBLE) / quantity), 4) AS avg_fill_rate,
                SUM(CASE WHEN is_anomalous THEN 1 ELSE 0 END) AS anomaly_count
            FROM orders
            GROUP BY CAST(decision_time AS DATE)
            ORDER BY trade_date
        """)

    def venue_market_share(self) -> pd.DataFrame:
        """Fill volume distribution by venue."""
        return self.query("""
            SELECT
                venue,
                COUNT(*) AS fill_count,
                SUM(fill_quantity) AS total_shares,
                ROUND(SUM(fill_quantity) * 100.0 /
                    (SELECT SUM(fill_quantity) FROM fills), 2) AS market_share_pct
            FROM fills
            GROUP BY venue
            ORDER BY total_shares DESC
        """)

    def is_decomposition_summary(self) -> pd.DataFrame:
        """Summary of IS components."""
        return self.query("""
            SELECT
                'Delay Cost' AS component,
                ROUND(AVG(delay_cost_bps), 2) AS avg_bps,
                ROUND(MEDIAN(delay_cost_bps), 2) AS median_bps
            FROM orders
            UNION ALL
            SELECT
                'Execution Cost',
                ROUND(AVG(execution_cost_bps), 2),
                ROUND(MEDIAN(execution_cost_bps), 2)
            FROM orders
            UNION ALL
            SELECT
                'Opportunity Cost',
                ROUND(AVG(opportunity_cost_bps), 2),
                ROUND(MEDIAN(opportunity_cost_bps), 2)
            FROM orders
            UNION ALL
            SELECT
                'Fixed Cost',
                ROUND(AVG(fixed_cost_bps), 2),
                ROUND(MEDIAN(fixed_cost_bps), 2)
            FROM orders
            UNION ALL
            SELECT
                'Total IS',
                ROUND(AVG(total_is_bps), 2),
                ROUND(MEDIAN(total_is_bps), 2)
            FROM orders
        """)
