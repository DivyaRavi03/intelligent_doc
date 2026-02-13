"""Unit tests for the graceful degradation module."""

from __future__ import annotations

from src.llm.degradation import ComponentType, FallbackResult, GracefulDegradation


class TestGracefulDegradation:
    """Tests for :class:`GracefulDegradation`."""

    def test_wrap_returns_success_on_normal_call(self) -> None:
        """Successful function calls should return success=True."""
        gd = GracefulDegradation()
        result = gd.wrap(ComponentType.REDIS, lambda: "ok", "fallback")
        assert isinstance(result, FallbackResult)
        assert result.success is True
        assert result.value == "ok"
        assert result.fallback_used is False
        assert result.error is None

    def test_wrap_returns_fallback_on_exception(self) -> None:
        """Failed function calls should return the fallback value."""
        gd = GracefulDegradation()

        def failing():
            raise RuntimeError("connection refused")

        result = gd.wrap(ComponentType.REDIS, failing, "default_value")
        assert result.success is False
        assert result.value == "default_value"
        assert result.fallback_used is True
        assert result.error == "connection refused"

    def test_wrap_tracks_failure_count(self) -> None:
        """Each failure should increment the component's failure count."""
        gd = GracefulDegradation()

        def failing():
            raise RuntimeError("error")

        gd.wrap(ComponentType.LLM, failing, None)
        gd.wrap(ComponentType.LLM, failing, None)
        gd.wrap(ComponentType.LLM, failing, None)

        health = gd.get_health()
        assert health["llm"]["failure_count"] == 3

    def test_get_health_reports_components(self) -> None:
        """Health should include all component types."""
        gd = GracefulDegradation()
        health = gd.get_health()
        for ct in ComponentType:
            assert ct.value in health
            assert health[ct.value]["failure_count"] == 0
            assert health[ct.value]["last_error"] is None

    def test_reset_clears_failures(self) -> None:
        """reset() should zero all failure counters."""
        gd = GracefulDegradation()

        def failing():
            raise RuntimeError("boom")

        gd.wrap(ComponentType.DATABASE, failing, None)
        assert gd.get_health()["database"]["failure_count"] == 1

        gd.reset()
        assert gd.get_health()["database"]["failure_count"] == 0
        assert gd.get_health()["database"]["last_error"] is None

    def test_fallback_can_be_callable(self) -> None:
        """Callable fallbacks should be invoked with the same args."""
        gd = GracefulDegradation()

        def failing(x, y):
            raise RuntimeError("fail")

        result = gd.wrap(
            ComponentType.VECTOR_STORE,
            failing,
            lambda x, y: x + y,
            3,
            4,
        )
        assert result.value == 7
        assert result.fallback_used is True

    def test_fallback_can_be_static_value(self) -> None:
        """Static fallback values should be returned as-is."""
        gd = GracefulDegradation()

        def failing():
            raise RuntimeError("fail")

        result = gd.wrap(ComponentType.REDIS, failing, [])
        assert result.value == []
        assert result.fallback_used is True

    def test_multiple_components_tracked_independently(self) -> None:
        """Failures in different components should be tracked separately."""
        gd = GracefulDegradation()

        def failing():
            raise RuntimeError("fail")

        gd.wrap(ComponentType.REDIS, failing, None)
        gd.wrap(ComponentType.REDIS, failing, None)
        gd.wrap(ComponentType.LLM, failing, None)

        health = gd.get_health()
        assert health["redis"]["failure_count"] == 2
        assert health["llm"]["failure_count"] == 1
        assert health["vector_store"]["failure_count"] == 0
        assert health["database"]["failure_count"] == 0
