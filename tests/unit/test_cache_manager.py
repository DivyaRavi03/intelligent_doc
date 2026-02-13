"""Unit tests for the multi-tier cache manager."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from src.llm.cache_manager import TASK_TTLS, CacheManager


class TestCacheManager:
    """Tests for :class:`CacheManager`."""

    def test_set_and_get_from_memory(self) -> None:
        """Values stored in memory should be retrievable."""
        cm = CacheManager()  # no Redis
        cm.set("key1", "value1")
        assert cm.get("key1") == "value1"

    def test_cache_miss_returns_none(self) -> None:
        """Missing keys should return None."""
        cm = CacheManager()
        assert cm.get("nonexistent") is None

    def test_make_key_deterministic(self) -> None:
        """Same inputs should produce the same key."""
        k1 = CacheManager.make_key("prompt", "model", "qa")
        k2 = CacheManager.make_key("prompt", "model", "qa")
        assert k1 == k2

    def test_make_key_different_for_different_inputs(self) -> None:
        """Different inputs should produce different keys."""
        k1 = CacheManager.make_key("prompt_a", "model", "qa")
        k2 = CacheManager.make_key("prompt_b", "model", "qa")
        assert k1 != k2

    def test_make_key_different_for_different_tasks(self) -> None:
        """Different task types should produce different keys."""
        k1 = CacheManager.make_key("prompt", "model", "qa")
        k2 = CacheManager.make_key("prompt", "model", "extraction")
        assert k1 != k2

    def test_task_ttl_defaults(self) -> None:
        """TASK_TTLS should have entries for known task types."""
        assert "extraction" in TASK_TTLS
        assert "qa" in TASK_TTLS
        assert "summarization" in TASK_TTLS
        assert "comparison" in TASK_TTLS
        for ttl in TASK_TTLS.values():
            assert ttl > 0

    def test_memory_expiry(self) -> None:
        """Expired entries should not be returned."""
        cm = CacheManager()
        # Manually insert with past expiry
        cm._memory["expired_key"] = ("old_value", time.time() - 10)
        assert cm.get("expired_key") is None
        assert "expired_key" not in cm._memory

    def test_get_stats_tracks_hits_misses(self) -> None:
        """Stats should accurately track hits and misses."""
        cm = CacheManager()
        cm.set("k1", "v1")
        cm.get("k1")  # hit
        cm.get("k2")  # miss

        stats = cm.get_stats()
        assert stats["memory_hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_clear_empties_cache(self) -> None:
        """clear() should empty the memory cache and reset stats."""
        cm = CacheManager()
        cm.set("k1", "v1")
        cm.get("k1")
        cm.clear()

        assert cm.get("k1") is None
        stats = cm.get_stats()
        assert stats["memory_hits"] == 0
        assert stats["memory_entries"] == 0

    def test_redis_unavailable_falls_back_to_memory(self) -> None:
        """When Redis is unavailable, memory cache still works."""
        cm = CacheManager(redis_url="redis://invalid:9999")
        cm.set("k1", "v1")
        assert cm.get("k1") == "v1"

    def test_get_stats_initial(self) -> None:
        """Fresh cache should have zero stats."""
        cm = CacheManager()
        stats = cm.get_stats()
        assert stats["redis_hits"] == 0
        assert stats["memory_hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0
        assert stats["memory_entries"] == 0
