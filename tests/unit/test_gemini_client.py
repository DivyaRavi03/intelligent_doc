"""Unit tests for the centralized Gemini API client."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from src.llm.gemini_client import GeminiClient
from src.models.schemas import LLMResponse


def _mock_response(text: str = "hello", input_tokens: int = 10, output_tokens: int = 5) -> MagicMock:
    """Helper to create a mock Gemini API response."""
    response = MagicMock()
    response.text = text
    usage = MagicMock()
    usage.prompt_token_count = input_tokens
    usage.candidates_token_count = output_tokens
    response.usage_metadata = usage
    return response


class TestGeminiClientGenerate:
    """Tests for :meth:`GeminiClient.generate`."""

    # ------------------------------------------------------------------
    # Basic generation
    # ------------------------------------------------------------------

    @patch("src.llm.gemini_client.settings")
    def test_generate_returns_llm_response(self, mock_settings: MagicMock) -> None:
        """Should return a well-formed LLMResponse on success."""
        mock_settings.gemini_api_key = "fake-key"
        mock_settings.gemini_model = "gemini-2.0-flash"

        client = GeminiClient()
        mock_resp = _mock_response('{"answer": "test"}', 10, 5)

        with patch.object(client, "_check_cache", return_value=None), \
             patch.object(client, "_store_cache"), \
             patch.object(client, "_retry_with_backoff", return_value=mock_resp):
            result = client.generate("test prompt")

        assert isinstance(result, LLMResponse)
        assert result.content == '{"answer": "test"}'
        assert result.input_tokens == 10
        assert result.output_tokens == 5
        assert result.cached is False
        assert result.cost_usd >= 0.0

    @patch("src.llm.gemini_client.settings")
    def test_generate_missing_api_key_raises(self, mock_settings: MagicMock) -> None:
        """Should raise RuntimeError when API key is not set."""
        mock_settings.gemini_api_key = ""

        client = GeminiClient()
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            client.generate("test")

    @patch("src.llm.gemini_client.settings")
    def test_generate_with_system_instruction(self, mock_settings: MagicMock) -> None:
        """Should work correctly when system_instruction is provided."""
        mock_settings.gemini_api_key = "fake-key"
        mock_settings.gemini_model = "gemini-2.0-flash"

        client = GeminiClient()
        mock_resp = _mock_response("result")

        with patch.object(client, "_check_cache", return_value=None), \
             patch.object(client, "_store_cache"), \
             patch.object(client, "_retry_with_backoff", return_value=mock_resp):
            result = client.generate("prompt", system_instruction="be concise")

        assert isinstance(result, LLMResponse)
        assert result.content == "result"

    @patch("src.llm.gemini_client.settings")
    def test_generate_includes_latency(self, mock_settings: MagicMock) -> None:
        """Should track latency in milliseconds."""
        mock_settings.gemini_api_key = "fake-key"
        mock_settings.gemini_model = "gemini-2.0-flash"

        client = GeminiClient()
        mock_resp = _mock_response("result")

        with patch.object(client, "_check_cache", return_value=None), \
             patch.object(client, "_store_cache"), \
             patch.object(client, "_retry_with_backoff", return_value=mock_resp):
            result = client.generate("prompt")

        assert result.latency_ms >= 0.0
        assert result.model == "gemini-2.0-flash"

    # ------------------------------------------------------------------
    # Retry logic
    # ------------------------------------------------------------------

    def test_retry_succeeds_after_transient_failure(self) -> None:
        """Should retry on transient errors and eventually succeed."""
        client = GeminiClient(max_retries=3)

        call_count = 0

        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                exc = Exception("rate limit")
                exc.status_code = 429
                raise exc
            return "success"

        with patch("src.llm.gemini_client.time.sleep"):
            result = client._retry_with_backoff(flaky_func, max_retries=3)

        assert result == "success"
        assert call_count == 3

    def test_retry_raises_after_max_attempts(self) -> None:
        """Should raise RuntimeError after exhausting retries."""
        client = GeminiClient(max_retries=2)

        def always_fail():
            exc = Exception("server error")
            exc.status_code = 500
            raise exc

        with patch("src.llm.gemini_client.time.sleep"), \
             pytest.raises(RuntimeError, match="failed after 2 retries"):
            client._retry_with_backoff(always_fail, max_retries=2)

    def test_retry_does_not_retry_non_transient(self) -> None:
        """Non-retryable errors should propagate immediately."""
        client = GeminiClient()

        def bad_request():
            exc = Exception("bad request")
            exc.status_code = 400
            raise exc

        with pytest.raises(Exception, match="bad request"):
            client._retry_with_backoff(bad_request, max_retries=3)

    def test_retry_exponential_backoff_delays(self) -> None:
        """Should use exponential backoff delays: 1s, 2s, 4s."""
        client = GeminiClient()
        delays: list[float] = []

        def always_fail():
            exc = Exception("error")
            exc.status_code = 503
            raise exc

        with patch("src.llm.gemini_client.time.sleep", side_effect=lambda d: delays.append(d)):
            with pytest.raises(RuntimeError):
                client._retry_with_backoff(always_fail, max_retries=3)

        assert delays == [1.0, 2.0, 4.0]

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def test_cache_hit_returns_cached_content(self) -> None:
        """Cache hit should return content without calling Gemini."""
        client = GeminiClient()

        with patch.object(client, "_check_cache", return_value="cached answer"), \
             patch("src.llm.gemini_client.settings") as mock_settings:
            mock_settings.gemini_api_key = "fake-key"
            result = client.generate("test prompt")

        assert result.content == "cached answer"
        assert result.cached is True
        assert result.cost_usd == 0.0
        assert result.input_tokens == 0

    def test_cache_miss_calls_api(self) -> None:
        """Cache miss should call the API and store result."""
        client = GeminiClient()
        mock_resp = _mock_response("fresh answer")

        with patch.object(client, "_check_cache", return_value=None), \
             patch.object(client, "_store_cache") as mock_store, \
             patch.object(client, "_retry_with_backoff", return_value=mock_resp), \
             patch("src.llm.gemini_client.settings") as mock_settings:
            mock_settings.gemini_api_key = "fake-key"
            mock_settings.gemini_model = "gemini-2.0-flash"

            result = client.generate("test prompt")

        assert result.content == "fresh answer"
        assert result.cached is False
        mock_store.assert_called_once()

    def test_cache_key_deterministic(self) -> None:
        """Same inputs should produce the same cache key."""
        client = GeminiClient()
        key1 = client._compute_cache_key("prompt", "system")
        key2 = client._compute_cache_key("prompt", "system")
        assert key1 == key2

    def test_cache_key_varies_with_model(self) -> None:
        """Different models should produce different cache keys."""
        client_a = GeminiClient(model_name="model-a")
        client_b = GeminiClient(model_name="model-b")
        key_a = client_a._compute_cache_key("prompt", None)
        key_b = client_b._compute_cache_key("prompt", None)
        assert key_a != key_b

    def test_redis_unavailable_degrades_gracefully(self) -> None:
        """Missing Redis should not raise errors."""
        client = GeminiClient()
        client._redis_checked = False
        client._redis = None

        # _get_redis() catches all exceptions, so inject a module that fails.
        mock_redis = MagicMock()
        mock_redis.from_url.side_effect = ConnectionError("no redis")

        with patch.dict(sys.modules, {"redis": mock_redis}):
            result = client._get_redis()

        assert result is None

    # ------------------------------------------------------------------
    # Cost computation
    # ------------------------------------------------------------------

    def test_compute_cost_zero_tokens(self) -> None:
        """Zero tokens should cost $0."""
        assert GeminiClient._compute_cost(0, 0) == 0.0

    def test_compute_cost_known_values(self) -> None:
        """Should match expected Flash pricing."""
        # 1M input @ $0.075, 1M output @ $0.30
        cost = GeminiClient._compute_cost(1_000_000, 1_000_000)
        assert abs(cost - 0.375) < 1e-6

    def test_compute_cost_small_tokens(self) -> None:
        """Should handle small token counts accurately."""
        # 100 input tokens, 50 output tokens
        cost = GeminiClient._compute_cost(100, 50)
        expected = (100 * 0.075 + 50 * 0.30) / 1_000_000
        assert abs(cost - expected) < 1e-10
