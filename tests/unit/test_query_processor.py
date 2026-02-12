"""Unit tests for the query processing pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.models.schemas import SectionType
from src.retrieval.hybrid_retriever import HybridRetriever, RankedResult
from src.retrieval.query_processor import (
    QueryProcessor,
    QueryResult,
    QueryType,
    _QUERY_TYPE_PARAMS,
)
from src.retrieval.reranker import GeminiReranker


def _ranked(chunk_id: str, rrf_score: float = 0.01) -> RankedResult:
    """Helper to create a minimal RankedResult."""
    return RankedResult(
        chunk_id=chunk_id,
        text=f"Text for {chunk_id}",
        paper_id="paper-001",
        section_type="introduction",
        rrf_score=rrf_score,
        final_score=rrf_score,
    )


class TestQueryType:
    """Tests for :class:`QueryType`."""

    def test_query_type_values(self) -> None:
        """Enum values should match expected strings."""
        assert QueryType.FACTUAL.value == "factual"
        assert QueryType.CONCEPTUAL.value == "conceptual"
        assert QueryType.COMPARISON.value == "comparison"
        assert QueryType.METADATA.value == "metadata"

    def test_query_type_from_string(self) -> None:
        """Should be constructible from string values."""
        assert QueryType("factual") == QueryType.FACTUAL
        assert QueryType("conceptual") == QueryType.CONCEPTUAL


class TestQueryProcessor:
    """Tests for :class:`QueryProcessor`."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        """Create processor with mocked dependencies."""
        self.mock_retriever = MagicMock(spec=HybridRetriever)
        self.mock_reranker = MagicMock(spec=GeminiReranker)

        self.mock_retriever.retrieve.return_value = [_ranked("c1"), _ranked("c2")]
        self.mock_reranker.rerank.return_value = [_ranked("c1")]

        self.processor = QueryProcessor(
            hybrid_retriever=self.mock_retriever,
            reranker=self.mock_reranker,
        )

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    @patch("src.retrieval.query_processor.settings")
    def test_classify_factual(self, mock_settings: MagicMock) -> None:
        """Factual query should be classified as FACTUAL."""
        mock_settings.gemini_api_key = "fake-key"
        mock_settings.gemini_model = "gemini-2.0-flash"

        with patch.object(
            self.processor, "_call_gemini", return_value='{"type": "factual"}'
        ):
            result = self.processor.classify_query("What is the F1 score?")
        assert result == QueryType.FACTUAL

    @patch("src.retrieval.query_processor.settings")
    def test_classify_conceptual(self, mock_settings: MagicMock) -> None:
        """Conceptual query should be classified as CONCEPTUAL."""
        mock_settings.gemini_api_key = "fake-key"
        mock_settings.gemini_model = "gemini-2.0-flash"

        with patch.object(
            self.processor, "_call_gemini", return_value='{"type": "conceptual"}'
        ):
            result = self.processor.classify_query("How does attention work?")
        assert result == QueryType.CONCEPTUAL

    @patch("src.retrieval.query_processor.settings")
    def test_classify_comparison(self, mock_settings: MagicMock) -> None:
        """Comparison query should be classified as COMPARISON."""
        mock_settings.gemini_api_key = "fake-key"
        mock_settings.gemini_model = "gemini-2.0-flash"

        with patch.object(
            self.processor, "_call_gemini", return_value='{"type": "comparison"}'
        ):
            result = self.processor.classify_query("Compare BERT and GPT")
        assert result == QueryType.COMPARISON

    @patch("src.retrieval.query_processor.settings")
    def test_classify_metadata(self, mock_settings: MagicMock) -> None:
        """Metadata query should be classified as METADATA."""
        mock_settings.gemini_api_key = "fake-key"
        mock_settings.gemini_model = "gemini-2.0-flash"

        with patch.object(
            self.processor, "_call_gemini", return_value='{"type": "metadata"}'
        ):
            result = self.processor.classify_query("Who are the authors?")
        assert result == QueryType.METADATA

    @patch("src.retrieval.query_processor.settings")
    def test_classify_fallback_on_failure(self, mock_settings: MagicMock) -> None:
        """Gemini failure should default to FACTUAL."""
        mock_settings.gemini_api_key = "fake-key"
        mock_settings.gemini_model = "gemini-2.0-flash"

        with patch.object(
            self.processor, "_call_gemini", side_effect=RuntimeError("API down")
        ):
            result = self.processor.classify_query("anything")
        assert result == QueryType.FACTUAL

    @patch("src.retrieval.query_processor.settings")
    def test_classify_no_api_key_returns_factual(self, mock_settings: MagicMock) -> None:
        """No API key should default to FACTUAL."""
        mock_settings.gemini_api_key = ""

        result = self.processor.classify_query("anything")
        assert result == QueryType.FACTUAL

    # ------------------------------------------------------------------
    # Query expansion
    # ------------------------------------------------------------------

    @patch("src.retrieval.query_processor.settings")
    def test_expand_includes_original(self, mock_settings: MagicMock) -> None:
        """Original query should always be in the output."""
        mock_settings.gemini_api_key = "fake-key"
        mock_settings.gemini_model = "gemini-2.0-flash"

        with patch.object(
            self.processor,
            "_call_gemini",
            return_value='{"queries": ["reformulation 1", "reformulation 2"]}',
        ):
            result = self.processor.expand_query("original query")

        assert result[0] == "original query"
        assert len(result) >= 2

    @patch("src.retrieval.query_processor.settings")
    def test_expand_returns_reformulations(self, mock_settings: MagicMock) -> None:
        """Should return additional queries from Gemini."""
        mock_settings.gemini_api_key = "fake-key"
        mock_settings.gemini_model = "gemini-2.0-flash"

        with patch.object(
            self.processor,
            "_call_gemini",
            return_value='{"queries": ["alt1", "alt2"]}',
        ):
            result = self.processor.expand_query("test")

        assert "alt1" in result
        assert "alt2" in result

    @patch("src.retrieval.query_processor.settings")
    def test_expand_fallback_on_failure(self, mock_settings: MagicMock) -> None:
        """Gemini failure should return just the original query."""
        mock_settings.gemini_api_key = "fake-key"
        mock_settings.gemini_model = "gemini-2.0-flash"

        with patch.object(
            self.processor, "_call_gemini", side_effect=RuntimeError("API down")
        ):
            result = self.processor.expand_query("original")

        assert result == ["original"]

    @patch("src.retrieval.query_processor.settings")
    def test_expand_no_api_key_returns_original(self, mock_settings: MagicMock) -> None:
        """No API key should return just the original query."""
        mock_settings.gemini_api_key = ""

        result = self.processor.expand_query("original")
        assert result == ["original"]

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    @patch("src.retrieval.query_processor.settings")
    def test_process_returns_query_result(self, mock_settings: MagicMock) -> None:
        """process() should return a QueryResult."""
        mock_settings.gemini_api_key = "fake-key"
        mock_settings.gemini_model = "gemini-2.0-flash"

        with patch.object(self.processor, "classify_query", return_value=QueryType.FACTUAL):
            with patch.object(self.processor, "expand_query", return_value=["test"]):
                result = self.processor.process("test")

        assert isinstance(result, QueryResult)
        assert result.query == "test"
        assert result.query_type == QueryType.FACTUAL

    @patch("src.retrieval.query_processor.settings")
    def test_process_factual_sets_correct_params(self, mock_settings: MagicMock) -> None:
        """Factual queries should use alpha=0.5, top_k=5."""
        mock_settings.gemini_api_key = "fake-key"
        mock_settings.gemini_model = "gemini-2.0-flash"

        with patch.object(self.processor, "classify_query", return_value=QueryType.FACTUAL):
            with patch.object(self.processor, "expand_query", return_value=["test"]):
                self.processor.process("test")

        self.mock_retriever.retrieve.assert_called_once()
        call_kwargs = self.mock_retriever.retrieve.call_args
        assert call_kwargs.kwargs.get("alpha") == 0.5
        assert call_kwargs.kwargs.get("top_k") == 5

    @patch("src.retrieval.query_processor.settings")
    def test_process_conceptual_sets_correct_params(self, mock_settings: MagicMock) -> None:
        """Conceptual queries should use alpha=0.8, top_k=10."""
        mock_settings.gemini_api_key = "fake-key"
        mock_settings.gemini_model = "gemini-2.0-flash"

        with patch.object(self.processor, "classify_query", return_value=QueryType.CONCEPTUAL):
            with patch.object(self.processor, "expand_query", return_value=["test"]):
                self.processor.process("test")

        call_kwargs = self.mock_retriever.retrieve.call_args
        assert call_kwargs.kwargs.get("alpha") == 0.8
        assert call_kwargs.kwargs.get("top_k") == 10

    @patch("src.retrieval.query_processor.settings")
    def test_process_metadata_skips_retrieval(self, mock_settings: MagicMock) -> None:
        """Metadata queries should skip retrieval and return metadata_answer."""
        mock_settings.gemini_api_key = "fake-key"
        mock_settings.gemini_model = "gemini-2.0-flash"

        with patch.object(self.processor, "classify_query", return_value=QueryType.METADATA):
            result = self.processor.process("Who are the authors?")

        assert result.query_type == QueryType.METADATA
        assert result.metadata_answer is not None
        assert result.results == []
        self.mock_retriever.retrieve.assert_not_called()

    @patch("src.retrieval.query_processor.settings")
    def test_process_calls_reranker(self, mock_settings: MagicMock) -> None:
        """Reranker should be called with candidates from retriever."""
        mock_settings.gemini_api_key = "fake-key"
        mock_settings.gemini_model = "gemini-2.0-flash"

        with patch.object(self.processor, "classify_query", return_value=QueryType.FACTUAL):
            with patch.object(self.processor, "expand_query", return_value=["test"]):
                self.processor.process("test")

        self.mock_reranker.rerank.assert_called_once()

    @patch("src.retrieval.query_processor.settings")
    def test_process_passes_filters(self, mock_settings: MagicMock) -> None:
        """paper_id and section_type should propagate through pipeline."""
        mock_settings.gemini_api_key = "fake-key"
        mock_settings.gemini_model = "gemini-2.0-flash"

        with patch.object(self.processor, "classify_query", return_value=QueryType.FACTUAL):
            with patch.object(self.processor, "expand_query", return_value=["test"]):
                self.processor.process(
                    "test", paper_id="paper-X", section_type=SectionType.METHODOLOGY
                )

        call_kwargs = self.mock_retriever.retrieve.call_args
        assert call_kwargs.kwargs.get("paper_id") == "paper-X"
        assert call_kwargs.kwargs.get("section_type") == SectionType.METHODOLOGY

    def test_process_empty_query_returns_early(self) -> None:
        """Empty query should return early with default result."""
        result = self.processor.process("")
        assert result.query_type == QueryType.FACTUAL
        assert result.results == []

    # ------------------------------------------------------------------
    # Cross-query merge
    # ------------------------------------------------------------------

    def test_retrieve_and_merge_deduplicates(self) -> None:
        """Same chunk from multiple expanded queries should not duplicate."""
        self.mock_retriever.retrieve.return_value = [_ranked("c1")]

        result = self.processor._retrieve_and_merge(
            queries=["q1", "q2"],
            top_k=10,
            alpha=0.7,
            paper_id=None,
            section_type=None,
        )

        chunk_ids = [r.chunk_id for r in result]
        assert chunk_ids.count("c1") == 1

    def test_retrieve_and_merge_boosts_overlapping(self) -> None:
        """Chunks in multiple queries should get higher scores than single-query chunks."""
        # c1 appears in both queries, c2 only in query 1, c3 only in query 2
        self.mock_retriever.retrieve.side_effect = [
            [_ranked("c1", 0.02), _ranked("c2", 0.01)],
            [_ranked("c1", 0.02), _ranked("c3", 0.01)],
        ]

        result = self.processor._retrieve_and_merge(
            queries=["q1", "q2"],
            top_k=10,
            alpha=0.7,
            paper_id=None,
            section_type=None,
        )

        # c1 should rank highest (appears in both queries)
        assert result[0].chunk_id == "c1"

    def test_retrieve_and_merge_single_query_no_merge(self) -> None:
        """Single query should bypass cross-query RRF merge."""
        self.mock_retriever.retrieve.return_value = [_ranked("c1")]

        result = self.processor._retrieve_and_merge(
            queries=["q1"],
            top_k=10,
            alpha=0.7,
            paper_id=None,
            section_type=None,
        )

        assert len(result) == 1
        assert result[0].chunk_id == "c1"
