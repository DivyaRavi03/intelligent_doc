"""Centralized Gemini API client with caching, retry, and usage tracking.

The :class:`GeminiClient` is the single entry point for all Gemini LLM
calls in Phase 4.  It provides Redis-based response caching, exponential
backoff retry for transient errors (429 / 500 / 503), Prometheus metrics,
and per-call cost tracking using Gemini Flash pricing.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable
from typing import Any

from src.config import settings
from src.models.schemas import LLMResponse

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BASE_BACKOFF = 1.0  # seconds; delays: 1s, 2s, 4s
_RETRYABLE_STATUS_CODES = {429, 500, 503}
_CACHE_TTL_SECONDS = 3600  # 1 hour
_FLASH_INPUT_COST_PER_M = 0.075  # $/1M input tokens
_FLASH_OUTPUT_COST_PER_M = 0.30  # $/1M output tokens

# Prometheus metric singletons (created lazily)
_METRICS_INITIALISED = False
_LLM_REQUESTS: Any = None
_LLM_INPUT_TOKENS: Any = None
_LLM_OUTPUT_TOKENS: Any = None
_LLM_LATENCY: Any = None


def _init_metrics() -> None:
    """Lazily initialise Prometheus metrics (once)."""
    global _METRICS_INITIALISED, _LLM_REQUESTS, _LLM_INPUT_TOKENS, _LLM_OUTPUT_TOKENS, _LLM_LATENCY
    if _METRICS_INITIALISED:
        return
    try:
        from prometheus_client import Counter, Histogram

        _LLM_REQUESTS = Counter(
            "llm_requests_total",
            "Total LLM API requests",
            ["model"],
        )
        _LLM_INPUT_TOKENS = Counter(
            "llm_input_tokens_total",
            "Total input tokens sent to LLM",
            ["model"],
        )
        _LLM_OUTPUT_TOKENS = Counter(
            "llm_output_tokens_total",
            "Total output tokens from LLM",
            ["model"],
        )
        _LLM_LATENCY = Histogram(
            "llm_latency_seconds",
            "LLM request latency in seconds",
            ["model"],
        )
    except ImportError:
        logger.debug("prometheus_client not installed, metrics disabled")
    _METRICS_INITIALISED = True


class GeminiClient:
    """Centralized Gemini API client with caching, retry, and usage tracking.

    Args:
        model_name: Gemini model identifier.  Defaults to
            ``settings.gemini_model``.
        max_retries: Maximum retries on transient failures (429/500/503).
        cache_ttl: TTL for Redis cache entries in seconds.
    """

    def __init__(
        self,
        model_name: str | None = None,
        max_retries: int = _MAX_RETRIES,
        cache_ttl: int = _CACHE_TTL_SECONDS,
        model_router: Any = None,
        cache_manager: Any = None,
    ) -> None:
        self._model_name = model_name or settings.gemini_model
        self._max_retries = max_retries
        self._cache_ttl = cache_ttl
        self._redis: Any = None  # lazy
        self._redis_checked = False
        self._model_router = model_router
        self._cache_manager = cache_manager

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        response_schema: dict | None = None,
    ) -> LLMResponse:
        """Generate a response from Gemini.

        Args:
            prompt: The prompt text.
            system_instruction: Optional system instruction for the model.
            response_schema: Optional JSON schema to constrain output.

        Returns:
            :class:`LLMResponse` with content, token counts, cost, and
            cache status.

        Raises:
            RuntimeError: If API key is missing or all retries are exhausted.
        """
        self._ensure_api_key()

        # Check cache
        cache_key = self._compute_cache_key(prompt, system_instruction)
        cached = self._check_cache(cache_key)
        if cached is not None:
            return LLMResponse(
                content=cached,
                input_tokens=0,
                output_tokens=0,
                latency_ms=0.0,
                model=self._model_name,
                cached=True,
                cost_usd=0.0,
            )

        # Lazy import
        import google.generativeai as genai

        genai.configure(api_key=settings.gemini_api_key)

        if system_instruction:
            model = genai.GenerativeModel(
                self._model_name, system_instruction=system_instruction
            )
        else:
            model = genai.GenerativeModel(self._model_name)

        gen_config_kwargs: dict[str, Any] = {
            "response_mime_type": "application/json",
            "temperature": 0.0,
        }
        if response_schema is not None:
            gen_config_kwargs["response_schema"] = response_schema

        gen_config = genai.GenerationConfig(**gen_config_kwargs)

        start = time.perf_counter()

        def _call() -> Any:
            return model.generate_content(prompt, generation_config=gen_config)

        response = self._retry_with_backoff(_call, self._max_retries)
        latency_ms = (time.perf_counter() - start) * 1000.0

        # Extract token counts
        usage = getattr(response, "usage_metadata", None)
        input_tokens = getattr(usage, "prompt_token_count", 0) or 0
        output_tokens = getattr(usage, "candidates_token_count", 0) or 0

        content = response.text
        cost = self._compute_cost(input_tokens, output_tokens)

        # Store in cache
        self._store_cache(cache_key, content)

        # Track metrics
        self._track_usage(self._model_name, input_tokens, output_tokens, latency_ms)

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=round(latency_ms, 2),
            model=self._model_name,
            cached=False,
            cost_usd=round(cost, 8),
        )

    def generate_for_task(
        self,
        prompt: str,
        task_type: str,
        system_instruction: str | None = None,
        response_schema: dict | None = None,
    ) -> LLMResponse:
        """Route to the appropriate model and cache tier based on task type.

        Falls back to :meth:`generate` when no router/cache_manager is
        configured.
        """
        original_model = self._model_name
        try:
            if self._model_router:
                from src.llm.model_router import TaskType

                self._model_name = self._model_router.route(TaskType(task_type))

            if self._cache_manager:
                cache_key = self._cache_manager.make_key(
                    prompt, self._model_name, task_type
                )
                cached = self._cache_manager.get(cache_key, task_type)
                if cached is not None:
                    return LLMResponse(
                        content=cached,
                        input_tokens=0,
                        output_tokens=0,
                        latency_ms=0.0,
                        model=self._model_name,
                        cached=True,
                        cost_usd=0.0,
                    )

            result = self.generate(prompt, system_instruction, response_schema)

            if self._cache_manager:
                self._cache_manager.set(cache_key, result.content, task_type)

            return result
        finally:
            self._model_name = original_model

    # ------------------------------------------------------------------
    # Retry
    # ------------------------------------------------------------------

    def _retry_with_backoff(
        self,
        func: Callable[[], Any],
        max_retries: int = _MAX_RETRIES,
    ) -> Any:
        """Execute *func* with exponential-backoff retry on transient errors.

        Args:
            func: Zero-argument callable to execute.
            max_retries: Maximum number of attempts.

        Returns:
            The return value of *func*.

        Raises:
            RuntimeError: If all retries are exhausted.
        """
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as exc:
                last_exc = exc
                # Check if this is a retryable error
                status_code = getattr(exc, "status_code", None) or getattr(
                    exc, "code", None
                )
                if status_code and int(status_code) not in _RETRYABLE_STATUS_CODES:
                    raise

                wait = _BASE_BACKOFF * (2**attempt)
                logger.warning(
                    "Gemini attempt %d/%d failed (%s), retrying in %.1fs",
                    attempt + 1,
                    max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)

        raise RuntimeError(
            f"Gemini call failed after {max_retries} retries: {last_exc}"
        ) from last_exc

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def _compute_cache_key(
        self, prompt: str, system_instruction: str | None
    ) -> str:
        """Compute a deterministic cache key from model + prompt + instruction."""
        raw = f"{self._model_name}:{prompt}:{system_instruction or ''}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _check_cache(self, prompt_hash: str) -> str | None:
        """Check Redis for a cached response.  Returns ``None`` on miss."""
        client = self._get_redis()
        if client is None:
            return None
        try:
            return client.get(f"llm_cache:{prompt_hash}")
        except Exception:
            logger.debug("Redis GET failed", exc_info=True)
            return None

    def _store_cache(self, prompt_hash: str, content: str) -> None:
        """Store a response in Redis with TTL."""
        client = self._get_redis()
        if client is None:
            return
        try:
            client.setex(f"llm_cache:{prompt_hash}", self._cache_ttl, content)
        except Exception:
            logger.debug("Redis SETEX failed", exc_info=True)

    def _get_redis(self) -> Any:
        """Lazily connect to Redis.  Returns ``None`` if unavailable."""
        if self._redis is not None:
            return self._redis
        if self._redis_checked:
            return None
        self._redis_checked = True
        try:
            import redis as redis_lib

            client = redis_lib.from_url(settings.redis_url, decode_responses=True)
            client.ping()
            self._redis = client
            return client
        except Exception:
            logger.warning("Redis unavailable, caching disabled")
            return None

    # ------------------------------------------------------------------
    # Usage tracking
    # ------------------------------------------------------------------

    def _track_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
    ) -> None:
        """Record usage in Prometheus metrics."""
        _init_metrics()
        if _LLM_REQUESTS is not None:
            _LLM_REQUESTS.labels(model=model).inc()
        if _LLM_INPUT_TOKENS is not None:
            _LLM_INPUT_TOKENS.labels(model=model).inc(input_tokens)
        if _LLM_OUTPUT_TOKENS is not None:
            _LLM_OUTPUT_TOKENS.labels(model=model).inc(output_tokens)
        if _LLM_LATENCY is not None:
            _LLM_LATENCY.labels(model=model).observe(latency_ms / 1000.0)

    @staticmethod
    def _compute_cost(input_tokens: int, output_tokens: int) -> float:
        """Compute cost in USD using Gemini Flash pricing."""
        return (
            input_tokens * _FLASH_INPUT_COST_PER_M
            + output_tokens * _FLASH_OUTPUT_COST_PER_M
        ) / 1_000_000

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_api_key() -> None:
        """Raise immediately if no API key is configured."""
        if not settings.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured. "
                "Set it in .env or as an environment variable."
            )
