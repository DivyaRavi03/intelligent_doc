"""Unit tests for the Gemini-based passage reranker."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.retrieval.hybrid_retriever import RankedResult
from src.retrieval.reranker import GeminiReranker


def _ranked(
    chunk_id: str,
    rrf_score: float = 0.01,
    text: str = "sample passage text",
) -> RankedResult:
    """Helper to create a minimal RankedResult."""
    return RankedResult(
        chunk_id=chunk_id,
        text=text,
        paper_id="paper-001",
        section_type="introduction",
        rrf_score=rrf_score,
        final_score=rrf_score,
    )


class TestGeminiReranker:
    """Tests for :class:`GeminiReranker`."""

    # ------------------------------------------------------------------
    # Score parsing
    # ------------------------------------------------------------------

    def test_parse_scores_valid_json(self) -> None:
        """Should parse standard {"scores": [...]} format."""
        reranker = GeminiReranker()
        scores = reranker._parse_scores('{"scores": [7, 3, 9]}', 3)
        assert scores == [7.0, 3.0, 9.0]

    def test_parse_scores_bare_list(self) -> None:
        """Should parse a bare JSON list."""
        reranker = GeminiReranker()
        scores = reranker._parse_scores("[7, 3, 9]", 3)
        assert scores == [7.0, 3.0, 9.0]

    def test_parse_scores_with_markdown_fences(self) -> None:
        """Should handle markdown code fences."""
        reranker = GeminiReranker()
        text = '```json\n{"scores": [7, 3]}\n```'
        scores = reranker._parse_scores(text, 2)
        assert scores == [7.0, 3.0]

    def test_parse_scores_invalid_json_returns_empty(self) -> None:
        """Invalid JSON should return empty list."""
        reranker = GeminiReranker()
        scores = reranker._parse_scores("not valid json", 3)
        assert scores == []

    def test_parse_scores_wrong_length_returns_empty(self) -> None:
        """Mismatched score count should return empty list."""
        reranker = GeminiReranker()
        scores = reranker._parse_scores('{"scores": [7, 3]}', 5)
        assert scores == []

    def test_parse_scores_clamps_out_of_range(self) -> None:
        """Scores outside 1-10 should be clamped."""
        reranker = GeminiReranker()
        scores = reranker._parse_scores('{"scores": [0, 15, 5]}', 3)
        assert scores == [1.0, 10.0, 5.0]

    # ------------------------------------------------------------------
    # Score normalisation
    # ------------------------------------------------------------------

    def test_normalize_score_1_gives_0(self) -> None:
        """Score 1 should normalise to 0.0."""
        assert GeminiReranker._normalize_score(1.0) == 0.0

    def test_normalize_score_10_gives_1(self) -> None:
        """Score 10 should normalise to 1.0."""
        assert GeminiReranker._normalize_score(10.0) == 1.0

    def test_normalize_score_5_point_5(self) -> None:
        """Score 5.5 should normalise to 0.5."""
        assert abs(GeminiReranker._normalize_score(5.5) - 0.5) < 1e-6

    # ------------------------------------------------------------------
    # Reranking
    # ------------------------------------------------------------------

    @patch("src.retrieval.reranker.settings")
    def test_rerank_updates_scores(self, mock_settings: MagicMock) -> None:
        """After reranking, rerank_score and final_score should be set."""
        mock_settings.gemini_api_key = "fake-key"
        mock_settings.gemini_model = "gemini-2.0-flash"

        reranker = GeminiReranker()
        candidates = [_ranked("c1", rrf_score=0.01), _ranked("c2", rrf_score=0.02)]

        with patch.object(reranker, "_call_gemini", return_value='{"scores": [8, 3]}'):
            results = reranker.rerank("test query", candidates, top_k=5)

        assert len(results) == 2
        for r in results:
            assert r.rerank_score > 0.0 or r.rerank_score == 0.0
            assert r.final_score > 0.0

    @patch("src.retrieval.reranker.settings")
    def test_rerank_combines_with_rrf_score(self, mock_settings: MagicMock) -> None:
        """final_score should be 0.7 * rerank + 0.3 * rrf."""
        mock_settings.gemini_api_key = "fake-key"
        mock_settings.gemini_model = "gemini-2.0-flash"

        reranker = GeminiReranker(rerank_weight=0.7, rrf_weight=0.3)
        candidates = [_ranked("c1", rrf_score=0.5)]

        with patch.object(reranker, "_call_gemini", return_value='{"scores": [10]}'):
            results = reranker.rerank("test", candidates, top_k=5)

        # rerank_score for 10 → normalised to 1.0
        # final = 0.7 * 1.0 + 0.3 * 0.5 = 0.85
        assert len(results) == 1
        assert abs(results[0].final_score - 0.85) < 0.01

    @patch("src.retrieval.reranker.settings")
    def test_rerank_respects_top_k(self, mock_settings: MagicMock) -> None:
        """Should return at most top_k results."""
        mock_settings.gemini_api_key = "fake-key"
        mock_settings.gemini_model = "gemini-2.0-flash"

        reranker = GeminiReranker()
        candidates = [_ranked(f"c{i}") for i in range(10)]
        scores_json = '{"scores": [' + ",".join(["5"] * 10) + "]}"

        with patch.object(reranker, "_call_gemini", return_value=scores_json):
            results = reranker.rerank("test", candidates, top_k=3)

        assert len(results) == 3

    @patch("src.retrieval.reranker.settings")
    def test_rerank_reorders_candidates(self, mock_settings: MagicMock) -> None:
        """Higher Gemini scores should move results up."""
        mock_settings.gemini_api_key = "fake-key"
        mock_settings.gemini_model = "gemini-2.0-flash"

        reranker = GeminiReranker()
        candidates = [
            _ranked("c1", rrf_score=0.01),  # Was first
            _ranked("c2", rrf_score=0.009),  # Was second
        ]

        # Give c2 a much higher rerank score than c1
        with patch.object(reranker, "_call_gemini", return_value='{"scores": [2, 10]}'):
            results = reranker.rerank("test", candidates, top_k=5)

        assert results[0].chunk_id == "c2"
        assert results[1].chunk_id == "c1"

    # ------------------------------------------------------------------
    # Graceful degradation
    # ------------------------------------------------------------------

    @patch("src.retrieval.reranker.settings")
    def test_rerank_no_api_key_returns_original(self, mock_settings: MagicMock) -> None:
        """Missing API key should return candidates in original order."""
        mock_settings.gemini_api_key = ""

        reranker = GeminiReranker()
        candidates = [_ranked("c1"), _ranked("c2")]
        results = reranker.rerank("test", candidates, top_k=5)

        assert [r.chunk_id for r in results] == ["c1", "c2"]

    @patch("src.retrieval.reranker.settings")
    def test_rerank_api_failure_returns_original(self, mock_settings: MagicMock) -> None:
        """Gemini failure should return candidates in original order."""
        mock_settings.gemini_api_key = "fake-key"
        mock_settings.gemini_model = "gemini-2.0-flash"

        reranker = GeminiReranker()
        candidates = [_ranked("c1"), _ranked("c2")]

        with patch.object(reranker, "_call_gemini", side_effect=RuntimeError("API down")):
            results = reranker.rerank("test", candidates, top_k=5)

        assert [r.chunk_id for r in results] == ["c1", "c2"]

    @patch("src.retrieval.reranker.settings")
    def test_rerank_parse_failure_returns_original(self, mock_settings: MagicMock) -> None:
        """Bad JSON response should return candidates unchanged."""
        mock_settings.gemini_api_key = "fake-key"
        mock_settings.gemini_model = "gemini-2.0-flash"

        reranker = GeminiReranker()
        candidates = [_ranked("c1"), _ranked("c2")]

        with patch.object(reranker, "_call_gemini", return_value="not json"):
            results = reranker.rerank("test", candidates, top_k=5)

        assert [r.chunk_id for r in results] == ["c1", "c2"]

    def test_rerank_empty_candidates_returns_empty(self) -> None:
        """Empty input should return empty output."""
        reranker = GeminiReranker()
        results = reranker.rerank("test", [], top_k=5)
        assert results == []

    def test_rerank_empty_query_returns_original(self) -> None:
        """Empty query should return candidates unchanged."""
        reranker = GeminiReranker()
        candidates = [_ranked("c1")]
        results = reranker.rerank("", candidates, top_k=5)
        assert len(results) == 1
        assert results[0].chunk_id == "c1"

    # ------------------------------------------------------------------
    # Markdown fence stripping
    # ------------------------------------------------------------------

    def test_strip_markdown_fences(self) -> None:
        """Should extract content from fences."""
        text = '```json\n{"scores": [1]}\n```'
        result = GeminiReranker._strip_markdown_fences(text)
        assert result == '{"scores": [1]}'

    def test_strip_markdown_fences_no_fences(self) -> None:
        """Text without fences should be returned as-is (stripped)."""
        text = '  {"scores": [1]}  '
        result = GeminiReranker._strip_markdown_fences(text)
        assert result == '{"scores": [1]}'
