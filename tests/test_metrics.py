"""Tests for Implementation Shortfall computation."""

import pytest

from src.execution.metrics import ISDecomposition, compute_is_single


class TestISDecomposition:
    """Tests for the IS decomposition formula."""

    def test_is_zero_when_perfect_execution(self) -> None:
        """IS should be ~0 when fill matches decision, no delay, fully filled."""
        result = compute_is_single(
            order_id="TEST-001",
            side="BUY",
            decision_price=100.0,
            arrival_price=100.0,  # No delay drift
            vwap_fill_price=100.0,  # Fill at arrival
            close_price=100.0,  # No opportunity cost
            total_shares=1000,
            filled_shares=1000,  # Fully filled
            total_commission=0.0,  # No costs
        )
        assert abs(result.total_is_bps) < 0.01
        assert abs(result.delay_cost_bps) < 0.01
        assert abs(result.execution_cost_bps) < 0.01
        assert abs(result.opportunity_cost_bps) < 0.01

    def test_is_positive_for_adverse_buy(self) -> None:
        """IS should be positive when buy fills above decision price."""
        result = compute_is_single(
            order_id="TEST-002",
            side="BUY",
            decision_price=100.0,
            arrival_price=100.05,  # Price moved up during delay
            vwap_fill_price=100.10,  # Filled even higher (market impact)
            close_price=100.20,
            total_shares=1000,
            filled_shares=1000,
            total_commission=5.0,
        )
        assert result.total_is_bps > 0
        assert result.delay_cost_bps > 0  # Price moved against us
        assert result.execution_cost_bps > 0  # Filled above arrival

    def test_is_components_sum_to_total(self) -> None:
        """delay + execution + opportunity + fixed = total IS."""
        result = compute_is_single(
            order_id="TEST-003",
            side="BUY",
            decision_price=100.0,
            arrival_price=100.02,
            vwap_fill_price=100.08,
            close_price=100.15,
            total_shares=1000,
            filled_shares=800,
            total_commission=4.0,
        )
        component_sum = (
            result.delay_cost_bps
            + result.execution_cost_bps
            + result.opportunity_cost_bps
            + result.fixed_cost_bps
        )
        assert abs(result.total_is_bps - component_sum) < 0.01

    def test_sell_order_adverse_direction(self) -> None:
        """For SELL, IS is positive when fill is below decision price."""
        result = compute_is_single(
            order_id="TEST-004",
            side="SELL",
            decision_price=100.0,
            arrival_price=99.95,  # Price dropped during delay (adverse for sell)
            vwap_fill_price=99.90,  # Filled even lower
            close_price=99.80,
            total_shares=1000,
            filled_shares=1000,
            total_commission=0.0,
        )
        # For a sell, selling lower than decision is adverse
        assert result.execution_cost_bps > 0

    def test_unfilled_creates_opportunity_cost(self) -> None:
        """Partial fills should produce non-zero opportunity cost."""
        result = compute_is_single(
            order_id="TEST-005",
            side="BUY",
            decision_price=100.0,
            arrival_price=100.0,
            vwap_fill_price=100.0,
            close_price=105.0,  # Price moved up significantly
            total_shares=1000,
            filled_shares=500,  # Only half filled
            total_commission=0.0,
        )
        # Missed 500 shares that went up $5
        assert result.opportunity_cost_bps > 0

    def test_zero_shares_returns_zero(self) -> None:
        """Edge case: zero total shares should return all zeros."""
        result = compute_is_single(
            order_id="TEST-006",
            side="BUY",
            decision_price=100.0,
            arrival_price=100.0,
            vwap_fill_price=100.0,
            close_price=100.0,
            total_shares=0,
            filled_shares=0,
            total_commission=0.0,
        )
        assert result.total_is_bps == 0

    def test_fixed_cost_always_positive(self) -> None:
        """Commission cost is always positive regardless of side."""
        for side in ["BUY", "SELL"]:
            result = compute_is_single(
                order_id=f"TEST-{side}",
                side=side,
                decision_price=100.0,
                arrival_price=100.0,
                vwap_fill_price=100.0,
                close_price=100.0,
                total_shares=1000,
                filled_shares=1000,
                total_commission=10.0,
            )
            assert result.fixed_cost_bps > 0

    def test_known_value_computation(self) -> None:
        """Verify against a hand-calculated example.

        BUY 1000 shares:
        - Decision: $100.00
        - Arrival: $100.10  (delay cost = 10bps on 1000 shares)
        - VWAP Fill: $100.20 (exec cost = 10bps on 800 filled shares)
        - Close: $100.50
        - Filled: 800, Unfilled: 200
        - Commission: $4.00

        Paper value = $100.00 * 1000 = $100,000

        Delay = (100.10 - 100.00) * 1000 / 100000 * 10000 = 10.0 bps
        Execution = (100.20 - 100.10) * 800 / 100000 * 10000 = 8.0 bps
        Opportunity = (100.50 - 100.00) * 200 / 100000 * 10000 = 10.0 bps
        Fixed = 4.00 / 100000 * 10000 = 0.4 bps
        Total = 28.4 bps
        """
        result = compute_is_single(
            order_id="KNOWN",
            side="BUY",
            decision_price=100.0,
            arrival_price=100.10,
            vwap_fill_price=100.20,
            close_price=100.50,
            total_shares=1000,
            filled_shares=800,
            total_commission=4.0,
        )
        assert abs(result.delay_cost_bps - 10.0) < 0.01
        assert abs(result.execution_cost_bps - 8.0) < 0.01
        assert abs(result.opportunity_cost_bps - 10.0) < 0.01
        assert abs(result.fixed_cost_bps - 0.4) < 0.01
        assert abs(result.total_is_bps - 28.4) < 0.01
