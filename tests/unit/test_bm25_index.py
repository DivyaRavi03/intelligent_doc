"""Unit tests for the BM25 sparse keyword index."""

from __future__ import annotations

import pytest

from src.models.schemas import EnrichedChunk, SectionType
from src.retrieval.bm25_index import BM25Index, _STOPWORDS
from src.retrieval.vector_store import SearchResult


def _make_chunk(
    chunk_id: str,
    text: str,
    paper_id: str = "paper-001",
    section_type: SectionType = SectionType.INTRODUCTION,
    chunk_index: int = 0,
) -> EnrichedChunk:
    """Helper to create a minimal EnrichedChunk."""
    return EnrichedChunk(
        chunk_id=chunk_id,
        text=text,
        token_count=len(text) // 4,
        section_type=section_type,
        section_title=None,
        page_numbers=[0],
        paper_id=paper_id,
        paper_title="Test Paper",
        chunk_index=chunk_index,
        total_chunks=1,
    )


class TestBM25Index:
    """Tests for :class:`BM25Index`."""

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    def test_tokenize_lowercases(self) -> None:
        """Tokens should be lowercased."""
        idx = BM25Index(remove_stopwords=False)
        tokens = idx._tokenize("Hello World DEEP")
        assert tokens == ["hello", "world", "deep"]

    def test_tokenize_removes_punctuation(self) -> None:
        """Punctuation should be stripped."""
        idx = BM25Index(remove_stopwords=False)
        tokens = idx._tokenize("hello, world! test.")
        assert tokens == ["hello", "world", "test"]

    def test_tokenize_removes_stopwords(self) -> None:
        """Stopwords should be filtered when enabled."""
        idx = BM25Index(remove_stopwords=True)
        tokens = idx._tokenize("the quick brown fox")
        assert "the" not in tokens
        assert "quick" in tokens
        assert "brown" in tokens
        assert "fox" in tokens

    def test_tokenize_without_stopword_removal(self) -> None:
        """Stopwords should be kept when disabled."""
        idx = BM25Index(remove_stopwords=False)
        tokens = idx._tokenize("the quick brown fox")
        assert "the" in tokens

    def test_tokenize_empty_string(self) -> None:
        """Empty string should produce empty token list."""
        idx = BM25Index()
        assert idx._tokenize("") == []

    def test_tokenize_all_stopwords(self) -> None:
        """Text of only stopwords should produce empty list."""
        idx = BM25Index(remove_stopwords=True)
        tokens = idx._tokenize("the is a an of")
        assert tokens == []

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def test_build_index_stores_chunks(self) -> None:
        """After build, chunk_count should match input size."""
        idx = BM25Index()
        chunks = [_make_chunk("c1", "deep learning"), _make_chunk("c2", "machine learning")]
        idx.build_index(chunks)
        assert idx.chunk_count() == 2

    def test_build_index_sets_is_built(self) -> None:
        """is_built should return True after build."""
        idx = BM25Index()
        assert not idx.is_built()
        idx.build_index([_make_chunk("c1", "hello")])
        assert idx.is_built()

    def test_build_index_empty_raises(self) -> None:
        """Empty chunk list should raise ValueError."""
        idx = BM25Index()
        with pytest.raises(ValueError, match="empty"):
            idx.build_index([])

    def test_build_index_replaces_previous(self) -> None:
        """Building twice should replace the old index."""
        idx = BM25Index()
        idx.build_index([_make_chunk("c1", "first")])
        assert idx.chunk_count() == 1

        idx.build_index([_make_chunk("c2", "second"), _make_chunk("c3", "third")])
        assert idx.chunk_count() == 2

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def test_search_returns_search_results(self) -> None:
        """Results should be SearchResult instances."""
        idx = BM25Index()
        idx.build_index([
            _make_chunk("c1", "deep learning for NLP"),
            _make_chunk("c2", "quantum computing physics"),
            _make_chunk("c3", "biology and chemistry research"),
        ])
        results = idx.search("deep learning")

        assert len(results) >= 1
        assert isinstance(results[0], SearchResult)

    def test_search_before_build_raises(self) -> None:
        """Searching before build should raise RuntimeError."""
        idx = BM25Index()
        with pytest.raises(RuntimeError, match="not been built"):
            idx.search("test query")

    def test_search_exact_match_ranked_high(self) -> None:
        """A chunk with exact query terms should rank higher."""
        idx = BM25Index()
        chunks = [
            _make_chunk("c1", "transformer attention mechanism overview"),
            _make_chunk("c2", "deep learning for natural language processing"),
            _make_chunk("c3", "transformer architecture with multi-head attention"),
        ]
        idx.build_index(chunks)
        results = idx.search("transformer attention")

        assert len(results) >= 1
        # Chunks mentioning "transformer" and "attention" should rank top
        top_ids = [r.chunk_id for r in results[:2]]
        assert "c1" in top_ids or "c3" in top_ids

    def test_search_respects_n_results(self) -> None:
        """Should return at most n_results items."""
        idx = BM25Index()
        chunks = [_make_chunk(f"c{i}", f"deep learning topic {i}") for i in range(10)]
        idx.build_index(chunks)
        results = idx.search("deep learning", n_results=3)
        assert len(results) <= 3

    def test_search_scores_normalized_0_to_1(self) -> None:
        """All scores should be in [0, 1]."""
        idx = BM25Index()
        chunks = [
            _make_chunk("c1", "deep learning NLP"),
            _make_chunk("c2", "computer vision CNN"),
        ]
        idx.build_index(chunks)
        results = idx.search("deep learning")
        for r in results:
            assert 0.0 <= r.score <= 1.0

    def test_search_top_result_score_is_1(self) -> None:
        """The top result should have normalised score 1.0."""
        idx = BM25Index()
        chunks = [
            _make_chunk("c1", "deep learning NLP"),
            _make_chunk("c2", "deep learning vision"),
            _make_chunk("c3", "quantum computing physics"),
        ]
        idx.build_index(chunks)
        results = idx.search("deep learning")
        assert len(results) >= 1
        assert results[0].score == 1.0

    def test_search_no_match_returns_empty(self) -> None:
        """Query with no overlapping terms should return empty."""
        idx = BM25Index()
        idx.build_index([_make_chunk("c1", "deep learning for NLP")])
        results = idx.search("quantum computing physics")
        assert results == []

    def test_search_empty_query_returns_empty(self) -> None:
        """Empty query should return empty list."""
        idx = BM25Index()
        idx.build_index([_make_chunk("c1", "some text")])
        results = idx.search("")
        assert results == []

    # ------------------------------------------------------------------
    # Metadata filters
    # ------------------------------------------------------------------

    def test_search_filter_by_paper_id(self) -> None:
        """paper_id filter should only return chunks from that paper."""
        idx = BM25Index()
        chunks = [
            _make_chunk("c1", "deep learning approach", paper_id="paper-A"),
            _make_chunk("c2", "deep learning method", paper_id="paper-B"),
            _make_chunk("c3", "quantum computing physics", paper_id="paper-C"),
        ]
        idx.build_index(chunks)
        results = idx.search("deep learning", paper_id="paper-A")
        assert len(results) == 1
        assert results[0].paper_id == "paper-A"

    def test_search_filter_by_section_type(self) -> None:
        """section_type filter should only return matching sections."""
        idx = BM25Index()
        chunks = [
            _make_chunk("c1", "deep learning method", section_type=SectionType.METHODOLOGY),
            _make_chunk("c2", "deep learning intro", section_type=SectionType.INTRODUCTION),
            _make_chunk("c3", "quantum computing physics", section_type=SectionType.RESULTS),
        ]
        idx.build_index(chunks)
        results = idx.search("deep learning", section_type=SectionType.METHODOLOGY)
        assert len(results) == 1
        assert results[0].section_type == "methodology"

    def test_search_combined_filters(self) -> None:
        """Both paper_id and section_type filters applied together."""
        idx = BM25Index()
        chunks = [
            _make_chunk("c1", "deep learning A intro", paper_id="A", section_type=SectionType.INTRODUCTION),
            _make_chunk("c2", "deep learning A method", paper_id="A", section_type=SectionType.METHODOLOGY),
            _make_chunk("c3", "deep learning B intro", paper_id="B", section_type=SectionType.INTRODUCTION),
            _make_chunk("c4", "quantum computing physics", paper_id="C", section_type=SectionType.RESULTS),
        ]
        idx.build_index(chunks)
        results = idx.search(
            "deep learning", paper_id="A", section_type=SectionType.INTRODUCTION
        )
        assert len(results) == 1
        assert results[0].paper_id == "A"
        assert results[0].section_type == "introduction"
