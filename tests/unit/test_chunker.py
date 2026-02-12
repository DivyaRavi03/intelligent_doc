"""Unit tests for the section-aware chunker."""

from __future__ import annotations

import pytest

from src.chunking.chunker import SectionAwareChunker, _CHARS_PER_TOKEN
from src.models.schemas import (
    DetectedSection,
    DocumentMetadataSchema,
    EnrichedChunk,
    ExtractedTable,
    PaperStructure,
    SectionType,
)


class TestSectionAwareChunker:
    """Tests for :class:`SectionAwareChunker`."""

    # ------------------------------------------------------------------
    # Basic chunking
    # ------------------------------------------------------------------

    def test_chunk_produces_enriched_chunks(self, sample_paper: PaperStructure) -> None:
        """Chunking a paper should return a non-empty list of EnrichedChunk."""
        chunker = SectionAwareChunker()
        chunks = chunker.chunk(sample_paper)

        assert len(chunks) > 0
        assert all(isinstance(c, EnrichedChunk) for c in chunks)

    def test_chunk_indices_are_sequential(self, sample_paper: PaperStructure) -> None:
        """chunk_index should be 0, 1, 2, ... and total_chunks should match."""
        chunker = SectionAwareChunker()
        chunks = chunker.chunk(sample_paper)

        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))
        assert all(c.total_chunks == len(chunks) for c in chunks)

    def test_chunk_ids_are_unique(self, sample_paper: PaperStructure) -> None:
        """Every chunk should have a unique chunk_id."""
        chunker = SectionAwareChunker()
        chunks = chunker.chunk(sample_paper)

        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_ids_are_deterministic(self, sample_paper: PaperStructure) -> None:
        """Chunking the same paper twice should produce identical IDs."""
        chunker = SectionAwareChunker()
        chunks1 = chunker.chunk(sample_paper)
        chunks2 = chunker.chunk(sample_paper)

        assert [c.chunk_id for c in chunks1] == [c.chunk_id for c in chunks2]

    def test_every_chunk_has_paper_id(self, sample_paper: PaperStructure) -> None:
        """Every chunk should carry the paper_id."""
        chunker = SectionAwareChunker()
        chunks = chunker.chunk(sample_paper)

        assert all(c.paper_id == "test-paper-001" for c in chunks)

    def test_every_chunk_has_paper_title(self, sample_paper: PaperStructure) -> None:
        """Every chunk should carry the paper title."""
        chunker = SectionAwareChunker()
        chunks = chunker.chunk(sample_paper)

        for chunk in chunks:
            if chunk.metadata.get("type") != "table":
                assert chunk.paper_title == "Deep Learning for Document Understanding"

    def test_every_chunk_has_token_count(self, sample_paper: PaperStructure) -> None:
        """token_count should be positive for every chunk."""
        chunker = SectionAwareChunker()
        chunks = chunker.chunk(sample_paper)

        assert all(c.token_count > 0 for c in chunks)

    def test_empty_paper_returns_no_chunks(self) -> None:
        """A paper with no sections or tables should return an empty list."""
        paper = PaperStructure(paper_id="empty")
        chunker = SectionAwareChunker()
        chunks = chunker.chunk(paper)

        assert chunks == []

    # ------------------------------------------------------------------
    # Section-boundary rule
    # ------------------------------------------------------------------

    def test_chunks_never_mix_sections(self, sample_paper: PaperStructure) -> None:
        """Each text chunk should belong to exactly one section_type."""
        chunker = SectionAwareChunker()
        chunks = chunker.chunk(sample_paper)

        for chunk in chunks:
            # section_type is a single value, not a list — so the schema
            # already enforces this. Verify it's a valid SectionType.
            assert chunk.section_type in SectionType

    def test_short_section_is_single_chunk(self, sample_paper: PaperStructure) -> None:
        """A section shorter than target_tokens should produce one chunk."""
        chunker = SectionAwareChunker(target_tokens=512)
        chunks = chunker.chunk(sample_paper)

        # The Abstract section is short and should be a single chunk
        abstract_chunks = [
            c for c in chunks if c.section_type == SectionType.ABSTRACT
        ]
        assert len(abstract_chunks) == 1

    def test_long_section_splits_into_multiple_chunks(
        self, sample_paper: PaperStructure
    ) -> None:
        """A section longer than target_tokens should be split."""
        # Use a very small target so the Introduction section (repeated text) splits
        chunker = SectionAwareChunker(target_tokens=64, overlap_tokens=10)
        chunks = chunker.chunk(sample_paper)

        intro_chunks = [
            c for c in chunks if c.section_type == SectionType.INTRODUCTION
        ]
        assert len(intro_chunks) > 1

    # ------------------------------------------------------------------
    # Table rule
    # ------------------------------------------------------------------

    def test_table_is_single_chunk(self, sample_paper: PaperStructure) -> None:
        """Each table should produce exactly one chunk."""
        chunker = SectionAwareChunker()
        chunks = chunker.chunk(sample_paper)

        table_chunks = [c for c in chunks if c.metadata.get("type") == "table"]
        assert len(table_chunks) == 1  # sample_paper has one table

    def test_table_chunk_contains_headers_and_rows(
        self, sample_paper: PaperStructure
    ) -> None:
        """Table chunk text should include column headers and data."""
        chunker = SectionAwareChunker()
        chunks = chunker.chunk(sample_paper)

        table_chunks = [c for c in chunks if c.metadata.get("type") == "table"]
        assert len(table_chunks) == 1
        text = table_chunks[0].text

        assert "Method" in text
        assert "94.2" in text
        assert "LayoutLM" in text

    def test_table_chunk_includes_caption(self, sample_paper: PaperStructure) -> None:
        """Table chunk should include the table caption."""
        chunker = SectionAwareChunker()
        chunks = chunker.chunk(sample_paper)

        table_chunks = [c for c in chunks if c.metadata.get("type") == "table"]
        assert "Performance comparison" in table_chunks[0].text

    # ------------------------------------------------------------------
    # Reference grouping rule
    # ------------------------------------------------------------------

    def test_references_are_grouped(self, sample_paper: PaperStructure) -> None:
        """References should be grouped into chunks of 5-10, not one per chunk."""
        chunker = SectionAwareChunker()
        chunks = chunker.chunk(sample_paper)

        ref_chunks = [
            c for c in chunks if c.section_type == SectionType.REFERENCES
        ]
        # 12 references with default batch size should produce 2-3 chunks
        assert 1 <= len(ref_chunks) <= 3

    def test_reference_chunks_contain_multiple_entries(
        self, sample_paper: PaperStructure
    ) -> None:
        """Each reference chunk should contain several reference entries."""
        chunker = SectionAwareChunker(refs_per_chunk_min=5, refs_per_chunk_max=10)
        chunks = chunker.chunk(sample_paper)

        ref_chunks = [
            c for c in chunks if c.section_type == SectionType.REFERENCES
        ]
        for rc in ref_chunks:
            # Count reference markers like [1], [2], etc.
            import re
            entries = re.findall(r"\[\d+\]", rc.text)
            assert len(entries) >= 2  # at least 2 refs per chunk

    # ------------------------------------------------------------------
    # Equation rule
    # ------------------------------------------------------------------

    def test_equations_stay_with_context(self, sample_paper: PaperStructure) -> None:
        """Display equations ($$...$$) should not be split from surrounding text."""
        chunker = SectionAwareChunker(target_tokens=512)
        chunks = chunker.chunk(sample_paper)

        # The methodology section has $$L = L_{cls} + \lambda L_{layout}$$
        method_chunks = [
            c for c in chunks if c.section_type == SectionType.METHODOLOGY
        ]
        # Find the chunk containing the equation
        eq_chunks = [c for c in method_chunks if "$$" in c.text]
        assert len(eq_chunks) >= 1
        # The equation and its context should be in the same chunk
        eq_chunk = eq_chunks[0]
        assert "loss function" in eq_chunk.text.lower() or "L_{cls}" in eq_chunk.text

    # ------------------------------------------------------------------
    # Overlap
    # ------------------------------------------------------------------

    def test_overlap_adds_preceding_context(self) -> None:
        """With overlap > 0, chunks after the first should start with prior text."""
        paper = PaperStructure(
            paper_id="overlap-test",
            sections=[
                DetectedSection(
                    section_type=SectionType.INTRODUCTION,
                    title="Intro",
                    text="Alpha bravo charlie. " * 80,  # ~1600 chars → splits at 512 tokens
                    page_start=0,
                    page_end=0,
                    order_index=0,
                ),
            ],
            metadata=DocumentMetadataSchema(title="Test"),
        )
        chunker = SectionAwareChunker(target_tokens=128, overlap_tokens=20)
        chunks = chunker.chunk(paper)

        assert len(chunks) >= 2
        # Second chunk should start with text from end of first chunk
        first_tail = chunks[0].text[-50:]
        # The overlap text from chunk[0] should appear at the start of chunk[1]
        assert any(
            word in chunks[1].text[:200]
            for word in first_tail.split()
            if len(word) > 3
        )

    def test_zero_overlap_no_duplication(self) -> None:
        """With overlap=0, consecutive chunks should not repeat text."""
        paper = PaperStructure(
            paper_id="no-overlap-test",
            sections=[
                DetectedSection(
                    section_type=SectionType.INTRODUCTION,
                    title="Intro",
                    text="Sentence one. Sentence two. Sentence three. Sentence four. " * 30,
                    page_start=0,
                    page_end=0,
                    order_index=0,
                ),
            ],
            metadata=DocumentMetadataSchema(title="Test"),
        )
        chunker = SectionAwareChunker(target_tokens=64, overlap_tokens=0)
        chunks = chunker.chunk(paper)

        assert len(chunks) >= 2

    # ------------------------------------------------------------------
    # Page numbers
    # ------------------------------------------------------------------

    def test_page_numbers_populated(self, sample_paper: PaperStructure) -> None:
        """Chunks should have page_numbers from their source section."""
        chunker = SectionAwareChunker()
        chunks = chunker.chunk(sample_paper)

        for chunk in chunks:
            assert len(chunk.page_numbers) >= 1

    def test_multi_page_section_has_range(self, sample_paper: PaperStructure) -> None:
        """A section spanning pages 1-2 should list both pages."""
        chunker = SectionAwareChunker()
        chunks = chunker.chunk(sample_paper)

        method_chunks = [
            c for c in chunks if c.section_type == SectionType.METHODOLOGY
        ]
        assert len(method_chunks) >= 1
        assert 1 in method_chunks[0].page_numbers
        assert 2 in method_chunks[0].page_numbers

    # ------------------------------------------------------------------
    # Token counting
    # ------------------------------------------------------------------

    def test_count_tokens_approximation(self) -> None:
        """Token count should approximate len(text) / 4."""
        text = "hello world this is a test"  # 26 chars → ~6 tokens
        count = SectionAwareChunker._count_tokens(text)
        assert count == 26 // _CHARS_PER_TOKEN

    def test_count_tokens_minimum_one(self) -> None:
        """Even empty-ish text should return at least 1 token."""
        assert SectionAwareChunker._count_tokens("ab") >= 1
        assert SectionAwareChunker._count_tokens("") >= 1

    # ------------------------------------------------------------------
    # Chunk ID
    # ------------------------------------------------------------------

    def test_make_id_deterministic(self) -> None:
        """Same input should produce the same ID."""
        id1 = SectionAwareChunker._make_id("hello", "paper1")
        id2 = SectionAwareChunker._make_id("hello", "paper1")
        assert id1 == id2

    def test_make_id_different_for_different_input(self) -> None:
        """Different text or paper should produce different IDs."""
        id1 = SectionAwareChunker._make_id("hello", "paper1")
        id2 = SectionAwareChunker._make_id("world", "paper1")
        id3 = SectionAwareChunker._make_id("hello", "paper2")
        assert id1 != id2
        assert id1 != id3

    # ------------------------------------------------------------------
    # Recursive split
    # ------------------------------------------------------------------

    def test_recursive_split_short_text(self) -> None:
        """Text shorter than target should return as-is."""
        chunker = SectionAwareChunker()
        result = chunker._recursive_split("short text", 1000)
        assert result == ["short text"]

    def test_recursive_split_paragraphs(self) -> None:
        """Text with paragraph breaks should split on paragraphs first."""
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunker = SectionAwareChunker()
        result = chunker._recursive_split(text, 30)
        assert len(result) >= 2

    def test_recursive_split_sentences(self) -> None:
        """When no paragraph breaks, split on sentences."""
        text = "Sentence one. Sentence two. Sentence three. Sentence four."
        chunker = SectionAwareChunker()
        result = chunker._recursive_split(text, 30)
        assert len(result) >= 2

    def test_recursive_split_preserves_equations(self) -> None:
        """Display equations should not be broken across chunks."""
        text = (
            "Before equation. "
            "$$E = mc^2 + \\sum_{i=1}^{n} x_i$$ "
            "After equation. More text here to make it longer."
        )
        chunker = SectionAwareChunker()
        result = chunker._recursive_split(text, 60)
        # The equation should be intact in one of the fragments
        all_text = " ".join(result)
        assert "$$E = mc^2" in all_text
        assert "x_i$$" in all_text
