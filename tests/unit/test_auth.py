"""Tests for API key authentication."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from src.api.auth import optional_api_key, verify_api_key
from src.api.stores import InMemoryAPIKeyStore


# ---------------------------------------------------------------------------
# verify_api_key
# ---------------------------------------------------------------------------


class TestVerifyAPIKey:
    """Tests for the verify_api_key dependency."""

    @pytest.fixture()
    def key_store(self) -> InMemoryAPIKeyStore:
        return InMemoryAPIKeyStore()

    @pytest.mark.asyncio
    async def test_valid_key(self, key_store: InMemoryAPIKeyStore) -> None:
        """Valid API key returns the key string."""
        result = await verify_api_key("test-api-key-12345", key_store)
        assert result == "test-api-key-12345"

    @pytest.mark.asyncio
    async def test_invalid_key(self, key_store: InMemoryAPIKeyStore) -> None:
        """Invalid key raises 401."""
        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key("bad-key", key_store)
        assert exc_info.value.status_code == 401
        assert "Invalid API key" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_missing_key_none(self, key_store: InMemoryAPIKeyStore) -> None:
        """Missing key (None) raises 401."""
        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(None, key_store)
        assert exc_info.value.status_code == 401
        assert "Missing" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_missing_key_empty(self, key_store: InMemoryAPIKeyStore) -> None:
        """Empty string key raises 401."""
        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key("", key_store)
        assert exc_info.value.status_code == 401
        assert "Missing" in exc_info.value.detail


# ---------------------------------------------------------------------------
# optional_api_key
# ---------------------------------------------------------------------------


class TestOptionalAPIKey:
    """Tests for the optional_api_key dependency."""

    @pytest.mark.asyncio
    async def test_returns_key(self) -> None:
        """Returns the key when present."""
        result = await optional_api_key("some-key")
        assert result == "some-key"

    @pytest.mark.asyncio
    async def test_returns_none(self) -> None:
        """Returns None when no key is provided."""
        result = await optional_api_key(None)
        assert result is None


# ---------------------------------------------------------------------------
# InMemoryAPIKeyStore
# ---------------------------------------------------------------------------


class TestAPIKeyStore:
    """Tests for the InMemoryAPIKeyStore."""

    def test_validate_known_key(self) -> None:
        """Pre-populated test key validates."""
        store = InMemoryAPIKeyStore()
        assert store.validate("test-api-key-12345") is True

    def test_validate_unknown_key(self) -> None:
        """Unknown key does not validate."""
        store = InMemoryAPIKeyStore()
        assert store.validate("unknown") is False

    def test_get_limits(self) -> None:
        """Returns rate limits for valid key."""
        store = InMemoryAPIKeyStore()
        limits = store.get_limits("test-api-key-12345")
        assert "uploads_per_hour" in limits
        assert "queries_per_hour" in limits

    def test_get_limits_unknown(self) -> None:
        """Returns empty dict for unknown key."""
        store = InMemoryAPIKeyStore()
        assert store.get_limits("unknown") == {}
