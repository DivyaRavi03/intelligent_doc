"""Unit tests for the ChromaDB vector store."""

from __future__ import annotations

import pytest

from src.models.schemas import EnrichedChunk, SectionType
from src.retrieval.vector_store import SearchResult, VectorStore


def _make_chunk(
    chunk_id: str,
    text: str,
    paper_id: str = "paper-001",
    section_type: SectionType = SectionType.INTRODUCTION,
    page_numbers: list[int] | None = None,
    chunk_index: int = 0,
) -> EnrichedChunk:
    """Helper to create a minimal EnrichedChunk."""
    return EnrichedChunk(
        chunk_id=chunk_id,
        text=text,
        token_count=len(text) // 4,
        section_type=section_type,
        section_title=None,
        page_numbers=page_numbers or [0],
        paper_id=paper_id,
        paper_title="Test Paper",
        chunk_index=chunk_index,
        total_chunks=1,
    )


def _fake_embedding(dim: int = 8, seed: float = 0.1) -> list[float]:
    """Generate a simple fake embedding vector."""
    return [seed + i * 0.01 for i in range(dim)]


class TestVectorStore:
    """Tests for :class:`VectorStore`."""

    @pytest.fixture(autouse=True)
    def _setup_store(self, tmp_path) -> None:
        """Create a fresh VectorStore with a temp directory for each test."""
        self.store = VectorStore(
            persist_dir=str(tmp_path / "chroma"),
            collection_name="test_chunks",
        )

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    def test_store_returns_count(self) -> None:
        """store() should return the number of chunks stored."""
        chunks = [_make_chunk("c1", "hello world")]
        embeddings = [_fake_embedding()]

        count = self.store.store(chunks, embeddings)
        assert count == 1

    def test_store_multiple_chunks(self) -> None:
        """Should store multiple chunks at once."""
        chunks = [
            _make_chunk("c1", "chunk one"),
            _make_chunk("c2", "chunk two"),
            _make_chunk("c3", "chunk three"),
        ]
        embeddings = [
            _fake_embedding(seed=0.1),
            _fake_embedding(seed=0.2),
            _fake_embedding(seed=0.3),
        ]

        count = self.store.store(chunks, embeddings)
        assert count == 3

    def test_store_empty_returns_zero(self) -> None:
        """Storing empty lists should return 0."""
        count = self.store.store([], [])
        assert count == 0

    def test_store_mismatched_lengths_raises(self) -> None:
        """Mismatched chunks and embeddings should raise ValueError."""
        chunks = [_make_chunk("c1", "text")]
        embeddings = [_fake_embedding(), _fake_embedding()]

        with pytest.raises(ValueError, match="Mismatch"):
            self.store.store(chunks, embeddings)

    def test_store_is_idempotent(self) -> None:
        """Upserting the same chunk twice should not create duplicates."""
        chunks = [_make_chunk("c1", "hello")]
        embeddings = [_fake_embedding()]

        self.store.store(chunks, embeddings)
        self.store.store(chunks, embeddings)

        results = self.store.search(_fake_embedding(), n_results=10)
        # Should still be only 1 chunk
        assert len(results) == 1

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def test_search_returns_results(self) -> None:
        """Search should find stored chunks."""
        chunks = [_make_chunk("c1", "deep learning for NLP")]
        embeddings = [_fake_embedding(seed=0.5)]

        self.store.store(chunks, embeddings)
        results = self.store.search(_fake_embedding(seed=0.5), n_results=5)

        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].chunk_id == "c1"
        assert results[0].text == "deep learning for NLP"

    def test_search_returns_similarity_score(self) -> None:
        """Results should have a score between 0 and 1."""
        chunks = [_make_chunk("c1", "test")]
        self.store.store(chunks, [_fake_embedding()])

        results = self.store.search(_fake_embedding(), n_results=5)
        assert len(results) == 1
        assert 0.0 <= results[0].score <= 1.0

    def test_search_respects_n_results(self) -> None:
        """Should return at most n_results items."""
        chunks = [_make_chunk(f"c{i}", f"text {i}") for i in range(10)]
        embeddings = [_fake_embedding(seed=0.1 * i) for i in range(10)]

        self.store.store(chunks, embeddings)
        results = self.store.search(_fake_embedding(), n_results=3)

        assert len(results) <= 3

    def test_search_empty_store_returns_empty(self) -> None:
        """Searching an empty store should return no results."""
        results = self.store.search(_fake_embedding(), n_results=5)
        assert results == []

    def test_search_metadata_populated(self) -> None:
        """Search results should carry metadata from the stored chunk."""
        chunks = [
            _make_chunk(
                "c1",
                "transformer attention",
                paper_id="paper-xyz",
                section_type=SectionType.METHODOLOGY,
                page_numbers=[2, 3],
            )
        ]
        self.store.store(chunks, [_fake_embedding()])

        results = self.store.search(_fake_embedding(), n_results=5)
        assert len(results) == 1
        assert results[0].paper_id == "paper-xyz"
        assert results[0].section_type == "methodology"
        assert results[0].page_numbers == [2, 3]

    # ------------------------------------------------------------------
    # Metadata filters
    # ------------------------------------------------------------------

    def test_search_filter_by_paper_id(self) -> None:
        """paper_id filter should only return chunks from that paper."""
        chunks = [
            _make_chunk("c1", "paper A text", paper_id="paper-A"),
            _make_chunk("c2", "paper B text", paper_id="paper-B"),
        ]
        embeddings = [_fake_embedding(seed=0.1), _fake_embedding(seed=0.2)]
        self.store.store(chunks, embeddings)

        results = self.store.search(
            _fake_embedding(), n_results=10, paper_id="paper-A"
        )
        assert len(results) == 1
        assert results[0].paper_id == "paper-A"

    def test_search_filter_by_section_type(self) -> None:
        """section_type filter should only return matching sections."""
        chunks = [
            _make_chunk("c1", "intro text", section_type=SectionType.INTRODUCTION),
            _make_chunk("c2", "method text", section_type=SectionType.METHODOLOGY),
        ]
        embeddings = [_fake_embedding(seed=0.1), _fake_embedding(seed=0.2)]
        self.store.store(chunks, embeddings)

        results = self.store.search(
            _fake_embedding(), n_results=10, section_type=SectionType.METHODOLOGY
        )
        assert len(results) == 1
        assert results[0].section_type == "methodology"

    def test_search_combined_filters(self) -> None:
        """Combined paper_id + section_type filter."""
        chunks = [
            _make_chunk("c1", "A intro", paper_id="A", section_type=SectionType.INTRODUCTION),
            _make_chunk("c2", "A methods", paper_id="A", section_type=SectionType.METHODOLOGY),
            _make_chunk("c3", "B intro", paper_id="B", section_type=SectionType.INTRODUCTION),
        ]
        embeddings = [_fake_embedding(seed=i * 0.1) for i in range(3)]
        self.store.store(chunks, embeddings)

        results = self.store.search(
            _fake_embedding(),
            n_results=10,
            paper_id="A",
            section_type=SectionType.INTRODUCTION,
        )
        assert len(results) == 1
        assert results[0].paper_id == "A"
        assert results[0].section_type == "introduction"

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def test_delete_paper_removes_chunks(self) -> None:
        """delete_paper should remove all chunks for that paper."""
        chunks = [
            _make_chunk("c1", "paper A chunk 1", paper_id="paper-A"),
            _make_chunk("c2", "paper A chunk 2", paper_id="paper-A"),
            _make_chunk("c3", "paper B chunk 1", paper_id="paper-B"),
        ]
        embeddings = [_fake_embedding(seed=i * 0.1) for i in range(3)]
        self.store.store(chunks, embeddings)

        deleted = self.store.delete_paper("paper-A")
        assert deleted == 2

        # Paper B should still be there
        results = self.store.search(_fake_embedding(), n_results=10)
        assert len(results) == 1
        assert results[0].paper_id == "paper-B"

    def test_delete_paper_nonexistent_returns_zero(self) -> None:
        """Deleting a paper with no chunks should return 0."""
        deleted = self.store.delete_paper("nonexistent-paper")
        assert deleted == 0

    def test_delete_then_search_returns_empty(self) -> None:
        """After deleting all chunks, search should return nothing."""
        chunks = [_make_chunk("c1", "only chunk", paper_id="paper-X")]
        self.store.store(chunks, [_fake_embedding()])

        self.store.delete_paper("paper-X")
        results = self.store.search(_fake_embedding(), n_results=10)
        assert results == []

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def test_build_where_no_filters(self) -> None:
        """No filters should return None."""
        assert VectorStore._build_where(None, None) is None

    def test_build_where_paper_only(self) -> None:
        """Single paper_id filter."""
        result = VectorStore._build_where("paper-1", None)
        assert result == {"paper_id": "paper-1"}

    def test_build_where_section_only(self) -> None:
        """Single section_type filter."""
        result = VectorStore._build_where(None, SectionType.ABSTRACT)
        assert result == {"section_type": "abstract"}

    def test_build_where_combined(self) -> None:
        """Combined filters should use $and."""
        result = VectorStore._build_where("paper-1", SectionType.ABSTRACT)
        assert result == {
            "$and": [
                {"paper_id": "paper-1"},
                {"section_type": "abstract"},
            ]
        }

    def test_chunk_to_metadata_flattens_correctly(self) -> None:
        """Metadata should be flat with string/int/float values."""
        chunk = _make_chunk("c1", "test", page_numbers=[1, 2, 3])
        meta = VectorStore._chunk_to_metadata(chunk)

        assert meta["paper_id"] == "paper-001"
        assert meta["section_type"] == "introduction"
        assert meta["page_numbers"] == "1,2,3"
        assert isinstance(meta["chunk_index"], int)
        assert isinstance(meta["token_count"], int)
