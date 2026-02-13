"""Multi-tier LLM response cache with task-aware TTLs.

L1 (Redis): fast shared cache with task-specific TTLs.
L2 (in-memory): local dict with expiry timestamps, used when Redis is
unavailable or as a fast-path fallback.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_TTL = 3600  # 1 hour

TASK_TTLS: dict[str, int] = {
    "extraction": 7200,
    "qa": 1800,
    "summarization": 3600,
    "comparison": 1800,
    "evaluation": 3600,
}


class CacheManager:
    """Two-tier LLM response cache (Redis L1 + in-memory L2).

    Args:
        redis_url: Redis connection URL.  ``None`` disables L1.
    """

    def __init__(self, redis_url: str | None = None) -> None:
        self._redis: Any = None
        self._redis_url = redis_url
        self._redis_checked = False

        # L2 in-memory: key → (value, expiry_timestamp)
        self._memory: dict[str, tuple[str, float]] = {}

        # Stats
        self._hits = {"redis": 0, "memory": 0}
        self._misses = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str, task_type: str = "") -> str | None:
        """Look up a cached value.  Checks L1 (Redis) then L2 (memory)."""
        # L1: Redis
        redis = self._get_redis()
        if redis is not None:
            try:
                val = redis.get(f"llm_cache:{key}")
                if val is not None:
                    self._hits["redis"] += 1
                    return val
            except Exception:
                logger.debug("Redis GET failed", exc_info=True)

        # L2: in-memory
        entry = self._memory.get(key)
        if entry is not None:
            value, expiry = entry
            if time.time() < expiry:
                self._hits["memory"] += 1
                return value
            # Expired — remove
            del self._memory[key]

        self._misses += 1
        return None

    def set(self, key: str, value: str, task_type: str = "") -> None:
        """Store a value in both cache tiers."""
        ttl = TASK_TTLS.get(task_type, _DEFAULT_TTL)

        # L1: Redis
        redis = self._get_redis()
        if redis is not None:
            try:
                redis.setex(f"llm_cache:{key}", ttl, value)
            except Exception:
                logger.debug("Redis SETEX failed", exc_info=True)

        # L2: in-memory
        self._memory[key] = (value, time.time() + ttl)

    @staticmethod
    def make_key(prompt: str, model: str, task_type: str = "") -> str:
        """Compute a deterministic cache key."""
        raw = f"{model}:{task_type}:{prompt}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get_stats(self) -> dict[str, Any]:
        """Return cache hit/miss statistics."""
        total = self._hits["redis"] + self._hits["memory"] + self._misses
        return {
            "redis_hits": self._hits["redis"],
            "memory_hits": self._hits["memory"],
            "misses": self._misses,
            "hit_rate": (
                (self._hits["redis"] + self._hits["memory"]) / total
                if total > 0
                else 0.0
            ),
            "memory_entries": len(self._memory),
        }

    def clear(self) -> None:
        """Clear all cache entries and reset stats."""
        self._memory.clear()
        self._hits = {"redis": 0, "memory": 0}
        self._misses = 0

        redis = self._get_redis()
        if redis is not None:
            try:
                # Only clear our namespace
                for k in redis.scan_iter("llm_cache:*"):
                    redis.delete(k)
            except Exception:
                logger.debug("Redis clear failed", exc_info=True)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_redis(self) -> Any:
        """Lazily connect to Redis.  Returns ``None`` if unavailable."""
        if self._redis is not None:
            return self._redis
        if self._redis_checked:
            return None
        self._redis_checked = True
        if not self._redis_url:
            return None
        try:
            import redis as redis_lib

            client = redis_lib.from_url(self._redis_url, decode_responses=True)
            client.ping()
            self._redis = client
            return client
        except Exception:
            logger.debug("Redis unavailable for CacheManager", exc_info=True)
            return None
