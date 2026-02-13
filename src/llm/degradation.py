"""Graceful degradation with component-specific fallback strategies.

Wraps function calls so that failures in non-critical components
(Redis, vector store) return safe fallback values instead of crashing
the request.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class ComponentType(StrEnum):
    LLM = "llm"
    REDIS = "redis"
    VECTOR_STORE = "vector_store"
    DATABASE = "database"


@dataclass
class FallbackResult:
    """Result of a wrapped call, indicating whether a fallback was used."""

    success: bool
    value: Any
    fallback_used: bool
    component: str
    error: str | None = None


@dataclass
class _ComponentState:
    failure_count: int = 0
    last_error: str | None = None


class GracefulDegradation:
    """Track component health and provide fallback values on failure."""

    def __init__(self) -> None:
        self._states: dict[str, _ComponentState] = {
            ct.value: _ComponentState() for ct in ComponentType
        }

    def wrap(
        self,
        component: ComponentType,
        func: Callable[..., Any],
        fallback: Callable[..., Any] | Any,
        *args: Any,
        **kwargs: Any,
    ) -> FallbackResult:
        """Execute *func* and return a :class:`FallbackResult`.

        On exception the fallback is used instead and the failure is
        recorded for the given component.
        """
        try:
            value = func(*args, **kwargs)
            return FallbackResult(
                success=True,
                value=value,
                fallback_used=False,
                component=component.value,
            )
        except Exception as exc:
            state = self._states[component.value]
            state.failure_count += 1
            state.last_error = str(exc)
            logger.warning(
                "%s failed (count=%d): %s", component.value, state.failure_count, exc
            )

            fb_value = fallback(*args, **kwargs) if callable(fallback) else fallback
            return FallbackResult(
                success=False,
                value=fb_value,
                fallback_used=True,
                component=component.value,
                error=str(exc),
            )

    def get_health(self) -> dict[str, Any]:
        """Return per-component failure counts and last errors."""
        return {
            name: {
                "failure_count": state.failure_count,
                "last_error": state.last_error,
            }
            for name, state in self._states.items()
        }

    def reset(self) -> None:
        """Reset all failure counters."""
        for state in self._states.values():
            state.failure_count = 0
            state.last_error = None
