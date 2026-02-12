"""In-memory LLM cost tracking with per-model and per-endpoint breakdown.

Uses the same thread-safe in-memory pattern as
:class:`src.api.stores.InMemoryDocumentStore`.  The interface is
designed to be swappable with a PostgreSQL-backed implementation.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

# Pricing per 1M tokens (USD)
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gemini-2.0-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
}

_DEFAULT_PRICING = {"input": 0.075, "output": 0.30}


@dataclass
class CostRecord:
    """A single LLM cost event."""

    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    endpoint: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class CostTracker:
    """Thread-safe in-memory LLM cost tracker.

    Args:
        pricing: Model pricing overrides.  Falls back to
            :data:`MODEL_PRICING` defaults.
    """

    def __init__(self, pricing: dict[str, dict[str, float]] | None = None) -> None:
        self._pricing = pricing or MODEL_PRICING
        self._records: list[CostRecord] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        endpoint: str = "",
    ) -> float:
        """Record token usage and return the computed cost in USD."""
        pricing = self._pricing.get(model, _DEFAULT_PRICING)
        cost = (
            input_tokens * pricing.get("input", 0.075)
            + output_tokens * pricing.get("output", 0.30)
        ) / 1_000_000

        record = CostRecord(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            endpoint=endpoint,
        )
        with self._lock:
            self._records.append(record)
        return cost

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_total_cost(self) -> float:
        """Return total cost across all records."""
        with self._lock:
            return sum(r.cost_usd for r in self._records)

    def get_daily_cost(self, day: date | None = None) -> float:
        """Return total cost for a specific day (default: today)."""
        target = day or date.today()
        with self._lock:
            return sum(
                r.cost_usd for r in self._records if r.timestamp.date() == target
            )

    def get_cost_by_model(self) -> dict[str, float]:
        """Return cost breakdown by model name."""
        breakdown: dict[str, float] = {}
        with self._lock:
            for r in self._records:
                breakdown[r.model] = breakdown.get(r.model, 0.0) + r.cost_usd
        return breakdown

    def get_cost_by_endpoint(self) -> dict[str, float]:
        """Return cost breakdown by endpoint."""
        breakdown: dict[str, float] = {}
        with self._lock:
            for r in self._records:
                key = r.endpoint or "unknown"
                breakdown[key] = breakdown.get(key, 0.0) + r.cost_usd
        return breakdown

    def get_summary(self) -> dict[str, Any]:
        """Return a full cost summary dict."""
        with self._lock:
            total_records = len(self._records)
        return {
            "total_cost": self.get_total_cost(),
            "daily_cost": self.get_daily_cost(),
            "by_model": self.get_cost_by_model(),
            "by_endpoint": self.get_cost_by_endpoint(),
            "total_records": total_records,
        }
