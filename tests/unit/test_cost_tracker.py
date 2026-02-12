"""Tests for the LLM cost tracker."""

from __future__ import annotations

from datetime import date

import pytest

from src.monitoring.cost_tracker import CostTracker, MODEL_PRICING


# ---------------------------------------------------------------------------
# Cost computation tests
# ---------------------------------------------------------------------------


class TestCostRecord:
    """Tests for record() and cost computation."""

    def test_flash_pricing(self) -> None:
        tracker = CostTracker()
        cost = tracker.record("gemini-2.0-flash", 1_000_000, 1_000_000)
        expected = (1_000_000 * 0.075 + 1_000_000 * 0.30) / 1_000_000
        assert abs(cost - expected) < 1e-6

    def test_zero_tokens(self) -> None:
        tracker = CostTracker()
        assert tracker.record("gemini-2.0-flash", 0, 0) == 0.0

    def test_pro_pricing(self) -> None:
        tracker = CostTracker()
        cost = tracker.record("gemini-1.5-pro", 1_000_000, 1_000_000)
        expected = (1_000_000 * 1.25 + 1_000_000 * 5.00) / 1_000_000
        assert abs(cost - expected) < 1e-6

    def test_unknown_model_uses_flash_fallback(self) -> None:
        tracker = CostTracker()
        cost = tracker.record("unknown-model", 1_000_000, 0)
        expected = 1_000_000 * 0.075 / 1_000_000
        assert abs(cost - expected) < 1e-6

    def test_small_token_counts(self) -> None:
        tracker = CostTracker()
        cost = tracker.record("gemini-2.0-flash", 100, 50)
        expected = (100 * 0.075 + 50 * 0.30) / 1_000_000
        assert abs(cost - expected) < 1e-10

    def test_record_returns_cost(self) -> None:
        tracker = CostTracker()
        result = tracker.record("gemini-2.0-flash", 500, 200)
        assert isinstance(result, float)
        assert result > 0.0

    def test_custom_pricing(self) -> None:
        custom = {"my-model": {"input": 1.0, "output": 2.0}}
        tracker = CostTracker(pricing=custom)
        cost = tracker.record("my-model", 1_000_000, 1_000_000)
        expected = (1_000_000 * 1.0 + 1_000_000 * 2.0) / 1_000_000
        assert abs(cost - expected) < 1e-6


# ---------------------------------------------------------------------------
# Aggregation tests
# ---------------------------------------------------------------------------


class TestAggregation:
    """Tests for get_total_cost, get_daily_cost, etc."""

    def test_get_total_cost(self) -> None:
        tracker = CostTracker()
        c1 = tracker.record("gemini-2.0-flash", 100, 50)
        c2 = tracker.record("gemini-2.0-flash", 200, 100)
        total = tracker.get_total_cost()
        assert abs(total - (c1 + c2)) < 1e-10

    def test_get_daily_cost_today(self) -> None:
        tracker = CostTracker()
        tracker.record("gemini-2.0-flash", 1000, 500)
        assert tracker.get_daily_cost() > 0

    def test_get_daily_cost_other_day(self) -> None:
        tracker = CostTracker()
        tracker.record("gemini-2.0-flash", 1000, 500)
        assert tracker.get_daily_cost(date(2020, 1, 1)) == 0.0

    def test_get_cost_by_model(self) -> None:
        tracker = CostTracker()
        tracker.record("gemini-2.0-flash", 1000, 500)
        tracker.record("gemini-1.5-pro", 1000, 500)
        breakdown = tracker.get_cost_by_model()
        assert "gemini-2.0-flash" in breakdown
        assert "gemini-1.5-pro" in breakdown
        assert breakdown["gemini-1.5-pro"] > breakdown["gemini-2.0-flash"]

    def test_get_cost_by_endpoint(self) -> None:
        tracker = CostTracker()
        tracker.record("gemini-2.0-flash", 1000, 500, endpoint="/api/v1/query")
        tracker.record("gemini-2.0-flash", 500, 250, endpoint="/api/v1/extract")
        breakdown = tracker.get_cost_by_endpoint()
        assert "/api/v1/query" in breakdown
        assert "/api/v1/extract" in breakdown

    def test_get_cost_by_endpoint_default(self) -> None:
        tracker = CostTracker()
        tracker.record("gemini-2.0-flash", 100, 50)
        breakdown = tracker.get_cost_by_endpoint()
        assert "unknown" in breakdown

    def test_get_summary(self) -> None:
        tracker = CostTracker()
        tracker.record("gemini-2.0-flash", 100, 50)
        summary = tracker.get_summary()
        assert "total_cost" in summary
        assert "daily_cost" in summary
        assert "by_model" in summary
        assert "by_endpoint" in summary
        assert summary["total_records"] == 1

    def test_empty_tracker(self) -> None:
        tracker = CostTracker()
        assert tracker.get_total_cost() == 0.0
        assert tracker.get_cost_by_model() == {}
        assert tracker.get_cost_by_endpoint() == {}


class TestModelPricing:
    """Tests for MODEL_PRICING constants."""

    def test_flash_pricing_exists(self) -> None:
        assert "gemini-2.0-flash" in MODEL_PRICING
        assert MODEL_PRICING["gemini-2.0-flash"]["input"] == 0.075
        assert MODEL_PRICING["gemini-2.0-flash"]["output"] == 0.30

    def test_pro_pricing_exists(self) -> None:
        assert "gemini-1.5-pro" in MODEL_PRICING
        assert MODEL_PRICING["gemini-1.5-pro"]["input"] == 1.25
        assert MODEL_PRICING["gemini-1.5-pro"]["output"] == 5.00
