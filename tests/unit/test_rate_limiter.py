"""Tests for rate limiter configuration."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.api.rate_limiter import (
    ADMIN_LIMIT,
    QUERY_LIMIT,
    UPLOAD_LIMIT,
    limiter,
    rate_limit_exceeded_handler,
)


class TestRateLimiterConfig:
    """Tests for rate limiter constants and handler."""

    def test_upload_limit(self) -> None:
        """Upload limit is set."""
        assert UPLOAD_LIMIT == "10/hour"

    def test_query_limit(self) -> None:
        """Query limit is set."""
        assert QUERY_LIMIT == "100/hour"

    def test_admin_limit(self) -> None:
        """Admin limit is set."""
        assert ADMIN_LIMIT == "20/minute"

    def test_limiter_exists(self) -> None:
        """Limiter instance is available."""
        assert limiter is not None

    def test_rate_limit_exceeded_handler_returns_429(self) -> None:
        """Handler returns a 429 response with Retry-After."""
        mock_request = MagicMock()
        mock_request.url.path = "/api/v1/query"
        mock_request.client.host = "127.0.0.1"
        mock_request.headers = {}

        exc = MagicMock()
        exc.detail = "100 per 1 hour"
        exc.retry_after = 3600

        response = rate_limit_exceeded_handler(mock_request, exc)

        assert response.status_code == 429
        assert "Retry-After" in response.headers
        assert response.headers["Retry-After"] == "3600"
        assert b"Rate limit exceeded" in response.body

    def test_rate_limit_handler_default_retry_after(self) -> None:
        """Handler defaults retry_after to 60 when not set."""
        mock_request = MagicMock()
        mock_request.url.path = "/test"
        mock_request.client.host = "127.0.0.1"
        mock_request.headers = {}

        exc = MagicMock()
        exc.detail = "limit exceeded"
        exc.retry_after = None

        response = rate_limit_exceeded_handler(mock_request, exc)

        assert response.status_code == 429
        assert response.headers["Retry-After"] == "60"
