"""Unit tests for the hybrid dense+sparse retriever."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.models.schemas import SectionType
from src.retrieval.bm25_index import BM25Index
from src.retrieval.embedding_service import EmbeddingService
from src.retrieval.hybrid_retriever import (
    HybridRetriever,
    RankedResult,
    _RRF_K,
)
from src.retrieval.vector_store import SearchResult, VectorStore


def _sr(chunk_id: str, score: float = 0.9, paper_id: str = "paper-001") -> SearchResult:
    """Helper to create a minimal SearchResult."""
    return SearchResult(
        chunk_id=chunk_id,
        text=f"Text for {chunk_id}",
        score=score,
        paper_id=paper_id,
        section_type="introduction",
    )


class TestRankedResult:
    """Tests for :class:`RankedResult`."""

    def test_default_scores(self) -> None:
        """Default scores should be 0.0."""
        r = RankedResult(chunk_id="c1", text="hello", paper_id="p1", section_type="intro")
        assert r.dense_score == 0.0
        assert r.sparse_score == 0.0
        assert r.rrf_score == 0.0
        assert r.rerank_score == 0.0
        assert r.final_score == 0.0

    def test_fields_stored(self) -> None:
        """All fields should be stored correctly."""
        r = RankedResult(
            chunk_id="c1", text="hello", paper_id="p1", section_type="intro",
            rrf_score=0.5, final_score=0.8,
        )
        assert r.chunk_id == "c1"
        assert r.rrf_score == 0.5
        assert r.final_score == 0.8


class TestHybridRetriever:
    """Tests for :class:`HybridRetriever`."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        """Create a retriever with mocked dependencies."""
        self.mock_vs = MagicMock(spec=VectorStore)
        self.mock_es = MagicMock(spec=EmbeddingService)
        self.mock_bm25 = MagicMock(spec=BM25Index)

        self.mock_es.embed_query.return_value = [0.1] * 8
        self.mock_bm25.is_built.return_value = True

        self.retriever = HybridRetriever(
            vector_store=self.mock_vs,
            embedding_service=self.mock_es,
            bm25_index=self.mock_bm25,
        )

    # ------------------------------------------------------------------
    # Fusion
    # ------------------------------------------------------------------

    def test_retrieve_returns_ranked_results(self) -> None:
        """Return type should be list of RankedResult."""
        self.mock_vs.search.return_value = [_sr("c1")]
        self.mock_bm25.search.return_value = [_sr("c1")]

        results = self.retriever.retrieve("test query")
        assert len(results) >= 1
        assert isinstance(results[0], RankedResult)

    def test_retrieve_combines_dense_and_sparse(self) -> None:
        """Both sources should contribute to the result set."""
        self.mock_vs.search.return_value = [_sr("c1"), _sr("c2")]
        self.mock_bm25.search.return_value = [_sr("c3"), _sr("c4")]

        results = self.retriever.retrieve("test query", top_k=10)
        result_ids = {r.chunk_id for r in results}
        assert "c1" in result_ids
        assert "c3" in result_ids

    def test_rrf_formula_correct(self) -> None:
        """Manually verify RRF scores for known ranks."""
        self.mock_vs.search.return_value = [_sr("c1")]  # rank 1
        self.mock_bm25.search.return_value = [_sr("c1")]  # rank 1

        results = self.retriever.retrieve("test", alpha=0.7)
        assert len(results) == 1

        k = _RRF_K
        expected = 0.7 * (1 / (k + 1)) + 0.3 * (1 / (k + 1))
        assert abs(results[0].rrf_score - round(expected, 6)) < 1e-5

    def test_missing_from_dense_gets_penalty_rank(self) -> None:
        """Chunk only in sparse should get rank=1000 for dense."""
        self.mock_vs.search.return_value = []
        self.mock_bm25.search.return_value = [_sr("c1")]  # rank 1 in sparse

        results = self.retriever.retrieve("test", alpha=0.7)
        assert len(results) == 1

        k = _RRF_K
        expected = 0.7 * (1 / (k + 1000)) + 0.3 * (1 / (k + 1))
        assert abs(results[0].rrf_score - round(expected, 6)) < 1e-5

    def test_missing_from_sparse_gets_penalty_rank(self) -> None:
        """Chunk only in dense should get rank=1000 for sparse."""
        self.mock_vs.search.return_value = [_sr("c1")]  # rank 1 in dense
        self.mock_bm25.search.return_value = []

        results = self.retriever.retrieve("test", alpha=0.7)
        assert len(results) == 1

        k = _RRF_K
        expected = 0.7 * (1 / (k + 1)) + 0.3 * (1 / (k + 1000))
        assert abs(results[0].rrf_score - round(expected, 6)) < 1e-5

    def test_duplicate_chunk_ids_handled(self) -> None:
        """Same chunk in both lists should be merged, not duplicated."""
        self.mock_vs.search.return_value = [_sr("c1")]
        self.mock_bm25.search.return_value = [_sr("c1")]

        results = self.retriever.retrieve("test")
        assert len(results) == 1

    def test_retrieve_sorted_by_rrf_score(self) -> None:
        """Results should be sorted by RRF score descending."""
        self.mock_vs.search.return_value = [_sr("c1"), _sr("c2")]
        self.mock_bm25.search.return_value = [_sr("c2"), _sr("c1")]

        results = self.retriever.retrieve("test", top_k=10)
        scores = [r.rrf_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_respects_top_k(self) -> None:
        """Should return at most top_k results."""
        self.mock_vs.search.return_value = [_sr(f"d{i}") for i in range(10)]
        self.mock_bm25.search.return_value = [_sr(f"s{i}") for i in range(10)]

        results = self.retriever.retrieve("test", top_k=3)
        assert len(results) <= 3

    # ------------------------------------------------------------------
    # Alpha override
    # ------------------------------------------------------------------

    def test_alpha_1_pure_dense(self) -> None:
        """With alpha=1.0, dense-only chunks should dominate."""
        self.mock_vs.search.return_value = [_sr("d1"), _sr("d2")]
        self.mock_bm25.search.return_value = [_sr("s1"), _sr("s2")]

        results = self.retriever.retrieve("test", alpha=1.0, top_k=10)
        # d1 is rank 1 in dense, rank 1000 in sparse → high score
        # s1 is rank 1000 in dense, rank 1 in sparse → low score (alpha=1.0 ignores sparse)
        d1_score = next(r.rrf_score for r in results if r.chunk_id == "d1")
        s1_score = next(r.rrf_score for r in results if r.chunk_id == "s1")
        assert d1_score > s1_score

    def test_alpha_0_pure_sparse(self) -> None:
        """With alpha=0.0, sparse-only chunks should dominate."""
        self.mock_vs.search.return_value = [_sr("d1"), _sr("d2")]
        self.mock_bm25.search.return_value = [_sr("s1"), _sr("s2")]

        results = self.retriever.retrieve("test", alpha=0.0, top_k=10)
        s1_score = next(r.rrf_score for r in results if r.chunk_id == "s1")
        d1_score = next(r.rrf_score for r in results if r.chunk_id == "d1")
        assert s1_score > d1_score

    def test_default_alpha_used_when_none(self) -> None:
        """When alpha is None, default should be used."""
        self.mock_vs.search.return_value = [_sr("c1")]
        self.mock_bm25.search.return_value = [_sr("c1")]

        # Default alpha is 0.7
        results = self.retriever.retrieve("test", alpha=None)
        assert len(results) == 1

        k = _RRF_K
        expected = 0.7 * (1 / (k + 1)) + 0.3 * (1 / (k + 1))
        assert abs(results[0].rrf_score - round(expected, 6)) < 1e-5

    # ------------------------------------------------------------------
    # Graceful degradation
    # ------------------------------------------------------------------

    def test_dense_failure_falls_back_to_sparse(self) -> None:
        """If embedding fails, sparse-only results should be returned."""
        self.mock_es.embed_query.side_effect = RuntimeError("API down")
        self.mock_bm25.search.return_value = [_sr("s1")]

        results = self.retriever.retrieve("test")
        assert len(results) >= 1
        assert results[0].chunk_id == "s1"

    def test_sparse_failure_falls_back_to_dense(self) -> None:
        """If BM25 not built, dense-only results should be returned."""
        self.mock_bm25.is_built.return_value = False
        self.mock_vs.search.return_value = [_sr("d1")]

        results = self.retriever.retrieve("test")
        assert len(results) >= 1
        assert results[0].chunk_id == "d1"

    def test_both_empty_returns_empty(self) -> None:
        """If both sources return empty, result is empty."""
        self.mock_vs.search.return_value = []
        self.mock_bm25.search.return_value = []

        results = self.retriever.retrieve("test")
        assert results == []

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def test_paper_id_passed_to_both_sources(self) -> None:
        """paper_id should be forwarded to both dense and sparse search."""
        self.mock_vs.search.return_value = []
        self.mock_bm25.search.return_value = []

        self.retriever.retrieve("test", paper_id="paper-X")

        self.mock_vs.search.assert_called_once()
        assert self.mock_vs.search.call_args.kwargs.get("paper_id") == "paper-X"
        self.mock_bm25.search.assert_called_once()
        assert self.mock_bm25.search.call_args.kwargs.get("paper_id") == "paper-X"

    def test_section_type_passed_to_both_sources(self) -> None:
        """section_type should be forwarded to both sources."""
        self.mock_vs.search.return_value = []
        self.mock_bm25.search.return_value = []

        self.retriever.retrieve("test", section_type=SectionType.METHODOLOGY)

        assert self.mock_vs.search.call_args.kwargs.get("section_type") == SectionType.METHODOLOGY
        assert self.mock_bm25.search.call_args.kwargs.get("section_type") == SectionType.METHODOLOGY
