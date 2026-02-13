"""Unit tests for the task-based model router."""

from __future__ import annotations

import pytest

from src.llm.model_router import ModelRouter, TaskType


class TestModelRouter:
    """Tests for :class:`ModelRouter`."""

    def test_default_routes_flash_for_extraction(self) -> None:
        """Extraction tasks should route to flash by default."""
        router = ModelRouter()
        assert router.route(TaskType.EXTRACTION) == "gemini-2.0-flash"

    def test_default_routes_flash_for_qa(self) -> None:
        """QA tasks should route to flash by default."""
        router = ModelRouter()
        assert router.route(TaskType.QA) == "gemini-2.0-flash"

    def test_default_routes_flash_for_summarization(self) -> None:
        """Summarization tasks should route to flash by default."""
        router = ModelRouter()
        assert router.route(TaskType.SUMMARIZATION) == "gemini-2.0-flash"

    def test_default_routes_pro_for_comparison(self) -> None:
        """Comparison tasks should route to pro by default."""
        router = ModelRouter()
        assert router.route(TaskType.COMPARISON) == "gemini-1.5-pro"

    def test_default_routes_pro_for_evaluation(self) -> None:
        """Evaluation tasks should route to pro by default."""
        router = ModelRouter()
        assert router.route(TaskType.EVALUATION) == "gemini-1.5-pro"

    def test_override_replaces_default(self) -> None:
        """Overrides should replace the default rule for a task type."""
        router = ModelRouter(overrides={TaskType.EXTRACTION: "gemini-1.5-pro"})
        assert router.route(TaskType.EXTRACTION) == "gemini-1.5-pro"
        # Other defaults still work
        assert router.route(TaskType.QA) == "gemini-2.0-flash"

    def test_all_task_types_have_routes(self) -> None:
        """Every TaskType should have a routing rule."""
        router = ModelRouter()
        for task in TaskType:
            result = router.route(task)
            assert isinstance(result, str)
            assert len(result) > 0

    def test_get_rules_returns_dict(self) -> None:
        """get_rules should return a dict with string keys and values."""
        router = ModelRouter()
        rules = router.get_rules()
        assert isinstance(rules, dict)
        assert len(rules) == len(TaskType)
        for key, val in rules.items():
            assert isinstance(key, str)
            assert isinstance(val, str)

    def test_route_returns_string(self) -> None:
        """route() should always return a string model name."""
        router = ModelRouter()
        result = router.route(TaskType.QA)
        assert isinstance(result, str)

    def test_invalid_task_type_raises(self) -> None:
        """Passing an invalid task type should raise ValueError."""
        with pytest.raises(ValueError):
            TaskType("nonexistent_task")
