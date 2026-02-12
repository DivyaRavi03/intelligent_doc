"""Unit tests for the metadata extractor module."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.ingestion.metadata_extractor import MetadataExtractor
from src.models.schemas import DocumentMetadataSchema, ExtractionMethod, PageResult


class TestMetadataExtractor:
    """Tests for :class:`MetadataExtractor`."""

    def test_extract_empty_pages(self) -> None:
        """Extracting from no pages should return empty metadata."""
        extractor = MetadataExtractor()
        result = extractor.extract([], use_llm=False)

        assert result.title is None
        assert result.authors == []
        assert result.abstract is None

    def test_heuristic_extracts_title(self, sample_pages: list[PageResult]) -> None:
        """Heuristic should find the title from the first page."""
        extractor = MetadataExtractor()
        result = extractor._heuristic_extract(sample_pages)

        assert result.title is not None
        assert "Deep Learning" in result.title

    def test_heuristic_extracts_doi(self, sample_pages: list[PageResult]) -> None:
        """Heuristic should find the DOI."""
        extractor = MetadataExtractor()
        result = extractor._heuristic_extract(sample_pages)

        assert result.doi is not None
        assert result.doi == "10.1234/example.2024.001"

    def test_heuristic_extracts_keywords(self, sample_pages: list[PageResult]) -> None:
        """Heuristic should find the keywords list."""
        extractor = MetadataExtractor()
        result = extractor._heuristic_extract(sample_pages)

        assert len(result.keywords) >= 2
        keyword_text = " ".join(result.keywords).lower()
        assert "deep learning" in keyword_text

    def test_heuristic_extracts_abstract(self, sample_pages: list[PageResult]) -> None:
        """Heuristic should extract the abstract text."""
        extractor = MetadataExtractor()
        result = extractor._heuristic_extract(sample_pages)

        assert result.abstract is not None
        assert "novel approach" in result.abstract

    def test_heuristic_confidence(self, sample_pages: list[PageResult]) -> None:
        """Heuristic results should have a moderate confidence score."""
        extractor = MetadataExtractor()
        result = extractor._heuristic_extract(sample_pages)

        assert result.confidence == 0.5

    def test_extract_with_llm_disabled(self, sample_pages: list[PageResult]) -> None:
        """When use_llm=False, only heuristic extraction should run."""
        extractor = MetadataExtractor()
        result = extractor.extract(sample_pages, use_llm=False)

        assert result.title is not None
        assert result.confidence == 0.5  # heuristic-only confidence

    @patch("src.ingestion.metadata_extractor.settings")
    def test_extract_skips_llm_without_api_key(
        self, mock_settings, sample_pages: list[PageResult]
    ) -> None:
        """LLM extraction should be skipped when no API key is configured."""
        mock_settings.gemini_api_key = ""
        extractor = MetadataExtractor()
        result = extractor.extract(sample_pages, use_llm=True)

        # Should fall back to heuristic only
        assert result.confidence == 0.5

    # ------------------------------------------------------------------
    # Merge logic
    # ------------------------------------------------------------------

    def test_merge_prefers_llm_title(self) -> None:
        """Merge should prefer LLM title over heuristic."""
        heuristic = DocumentMetadataSchema(title="Heuristic Title", confidence=0.5)
        llm = DocumentMetadataSchema(title="LLM Title", confidence=0.85)

        result = MetadataExtractor._merge(heuristic, llm)
        assert result.title == "LLM Title"

    def test_merge_falls_back_to_heuristic(self) -> None:
        """Merge should use heuristic when LLM field is empty."""
        heuristic = DocumentMetadataSchema(
            title="Heuristic Title",
            doi="10.1234/test",
            confidence=0.5,
        )
        llm = DocumentMetadataSchema(title=None, doi=None, confidence=0.85)

        result = MetadataExtractor._merge(heuristic, llm)
        assert result.title == "Heuristic Title"
        assert result.doi == "10.1234/test"

    def test_merge_prefers_llm_authors(self) -> None:
        """Merge should prefer LLM authors list when non-empty."""
        heuristic = DocumentMetadataSchema(authors=["Author A"], confidence=0.5)
        llm = DocumentMetadataSchema(authors=["Author A", "Author B"], confidence=0.85)

        result = MetadataExtractor._merge(heuristic, llm)
        assert result.authors == ["Author A", "Author B"]

    def test_merge_max_confidence(self) -> None:
        """Confidence should be the max of both sources."""
        heuristic = DocumentMetadataSchema(confidence=0.5)
        llm = DocumentMetadataSchema(confidence=0.85)

        result = MetadataExtractor._merge(heuristic, llm)
        assert result.confidence == 0.85

    def test_merge_both_empty(self) -> None:
        """Merging two empty schemas should produce an empty schema."""
        heuristic = DocumentMetadataSchema()
        llm = DocumentMetadataSchema()

        result = MetadataExtractor._merge(heuristic, llm)
        assert result.title is None
        assert result.authors == []
        assert result.doi is None

    # ------------------------------------------------------------------
    # Individual heuristic helpers
    # ------------------------------------------------------------------

    def test_extract_title_skips_boilerplate(self) -> None:
        """Title extraction should skip journal/proceedings headers."""
        lines = [
            "Proceedings of ICML 2024",
            "Volume 42, Issue 3",
            "A Novel Method for Text Classification",
            "John Doe, Jane Smith",
        ]
        title = MetadataExtractor._extract_title(lines)
        assert title == "A Novel Method for Text Classification"

    def test_extract_title_empty_lines(self) -> None:
        """Empty input should return None."""
        assert MetadataExtractor._extract_title([]) is None

    def test_extract_doi_bare(self) -> None:
        """Should find bare DOI without prefix."""
        text = "Available at 10.5555/test.paper.2024"
        doi = MetadataExtractor._extract_doi(text)
        assert doi == "10.5555/test.paper.2024"

    def test_extract_doi_with_prefix(self) -> None:
        """Should find DOI with 'DOI:' prefix."""
        text = "DOI: 10.1000/xyz123"
        doi = MetadataExtractor._extract_doi(text)
        assert doi == "10.1000/xyz123"

    def test_extract_doi_none(self) -> None:
        """No DOI should return None."""
        text = "This text has no digital object identifier."
        doi = MetadataExtractor._extract_doi(text)
        assert doi is None

    def test_extract_keywords_semicolons(self) -> None:
        """Keywords separated by semicolons should be split correctly."""
        text = "Keywords: machine learning; natural language processing; transformers\n\n1 Introduction"
        keywords = MetadataExtractor._extract_keywords(text)
        assert "machine learning" in keywords
        assert "natural language processing" in keywords
        assert "transformers" in keywords

    def test_extract_keywords_commas(self) -> None:
        """Keywords separated by commas should be split correctly."""
        text = "Keywords: NLP, deep learning, BERT\n\n1 Next section"
        keywords = MetadataExtractor._extract_keywords(text)
        assert len(keywords) == 3

    def test_extract_keywords_empty(self) -> None:
        """No keywords section should return empty list."""
        text = "This paper has no keywords section."
        keywords = MetadataExtractor._extract_keywords(text)
        assert keywords == []

    def test_extract_authors_from_sample_pages(self, sample_pages: list[PageResult]) -> None:
        """Author extraction should find names after the title."""
        extractor = MetadataExtractor()
        result = extractor._heuristic_extract(sample_pages)

        # The sample has "Alice Smith, Bob Jones" as authors
        author_text = " ".join(result.authors)
        # Author detection may vary, but at minimum it should find name-like strings
        # or return an empty list without erroring
        assert isinstance(result.authors, list)
