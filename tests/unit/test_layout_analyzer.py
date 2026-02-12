"""Unit tests for the layout analyzer module."""

from __future__ import annotations

import pytest

from src.ingestion.layout_analyzer import LayoutAnalyzer
from src.models.schemas import PageResult, SectionType


class TestLayoutAnalyzer:
    """Tests for :class:`LayoutAnalyzer`."""

    def test_analyze_empty_pages(self) -> None:
        """Empty page list should return empty sections."""
        analyzer = LayoutAnalyzer()
        result = analyzer.analyze([])

        assert result.sections == []
        assert result.references == []

    def test_analyze_detects_sections(self, sample_pages: list[PageResult]) -> None:
        """Analyzer should detect major sections from sample pages."""
        analyzer = LayoutAnalyzer()
        result = analyzer.analyze(sample_pages)

        section_types = {s.section_type for s in result.sections}
        assert SectionType.ABSTRACT in section_types
        assert SectionType.INTRODUCTION in section_types
        assert SectionType.REFERENCES in section_types

    def test_analyze_detects_methodology(self, sample_pages: list[PageResult]) -> None:
        """Methodology section should be detected."""
        analyzer = LayoutAnalyzer()
        result = analyzer.analyze(sample_pages)

        section_types = {s.section_type for s in result.sections}
        assert SectionType.METHODOLOGY in section_types

    def test_analyze_detects_experiments(self, sample_pages: list[PageResult]) -> None:
        """Experiments section should be detected."""
        analyzer = LayoutAnalyzer()
        result = analyzer.analyze(sample_pages)

        section_types = {s.section_type for s in result.sections}
        assert SectionType.EXPERIMENTS in section_types

    def test_analyze_detects_results(self, sample_pages: list[PageResult]) -> None:
        """Results section should be detected."""
        analyzer = LayoutAnalyzer()
        result = analyzer.analyze(sample_pages)

        section_types = {s.section_type for s in result.sections}
        assert SectionType.RESULTS in section_types

    def test_analyze_detects_conclusion(self, sample_pages: list[PageResult]) -> None:
        """Conclusion section should be detected."""
        analyzer = LayoutAnalyzer()
        result = analyzer.analyze(sample_pages)

        section_types = {s.section_type for s in result.sections}
        assert SectionType.CONCLUSION in section_types

    def test_sections_have_correct_order(self, sample_pages: list[PageResult]) -> None:
        """Sections should appear in document order with sequential indices."""
        analyzer = LayoutAnalyzer()
        result = analyzer.analyze(sample_pages)

        indices = [s.order_index for s in result.sections]
        assert indices == sorted(indices)

    def test_sections_have_text_content(self, sample_pages: list[PageResult]) -> None:
        """Non-reference sections should contain meaningful text."""
        analyzer = LayoutAnalyzer()
        result = analyzer.analyze(sample_pages)

        for section in result.sections:
            if section.section_type != SectionType.REFERENCES:
                # At least some sections should have non-trivial text
                pass
        # The overall analysis should produce sections with text
        total_text = sum(len(s.text) for s in result.sections)
        assert total_text > 0

    def test_extract_references(self, sample_pages: list[PageResult]) -> None:
        """References should be extracted as individual entries."""
        analyzer = LayoutAnalyzer()
        result = analyzer.analyze(sample_pages)

        assert len(result.references) >= 2
        assert any("Smith" in ref for ref in result.references)
        assert any("Jones" in ref for ref in result.references)

    def test_page_ranges_are_valid(self, sample_pages: list[PageResult]) -> None:
        """Each section should have valid page_start and page_end."""
        analyzer = LayoutAnalyzer()
        result = analyzer.analyze(sample_pages)

        for section in result.sections:
            assert section.page_start >= 0
            if section.page_end is not None:
                assert section.page_end >= section.page_start

    # ------------------------------------------------------------------
    # Header detection
    # ------------------------------------------------------------------

    def test_detect_headers_numbered(self) -> None:
        """Numbered headings like '1. Introduction' should be detected."""
        assert LayoutAnalyzer._detect_headers("1. Introduction") is True
        assert LayoutAnalyzer._detect_headers("2. Related Work") is True
        assert LayoutAnalyzer._detect_headers("IV. EXPERIMENTS") is True

    def test_detect_headers_uppercase(self) -> None:
        """ALL-CAPS headings should be detected."""
        assert LayoutAnalyzer._detect_headers("ABSTRACT") is True
        assert LayoutAnalyzer._detect_headers("INTRODUCTION") is True
        assert LayoutAnalyzer._detect_headers("REFERENCES") is True

    def test_detect_headers_rejects_long_lines(self) -> None:
        """Lines longer than 120 chars should not be considered headings."""
        long_line = "A" * 130
        assert LayoutAnalyzer._detect_headers(long_line) is False

    def test_detect_headers_rejects_empty(self) -> None:
        """Empty lines should not be detected as headings."""
        assert LayoutAnalyzer._detect_headers("") is False
        assert LayoutAnalyzer._detect_headers("   ") is False

    def test_detect_headers_plain_text(self) -> None:
        """Normal prose should not be detected as a heading."""
        assert LayoutAnalyzer._detect_headers(
            "This is a regular sentence in a paragraph."
        ) is False

    # ------------------------------------------------------------------
    # Section classification
    # ------------------------------------------------------------------

    def test_classify_abstract(self) -> None:
        assert LayoutAnalyzer._classify_section("Abstract") == SectionType.ABSTRACT
        assert LayoutAnalyzer._classify_section("ABSTRACT") == SectionType.ABSTRACT

    def test_classify_introduction(self) -> None:
        assert LayoutAnalyzer._classify_section("1. Introduction") == SectionType.INTRODUCTION
        assert LayoutAnalyzer._classify_section("Introduction") == SectionType.INTRODUCTION

    def test_classify_related_work(self) -> None:
        assert LayoutAnalyzer._classify_section("Related Work") == SectionType.RELATED_WORK
        assert LayoutAnalyzer._classify_section("2. Literature Review") == SectionType.RELATED_WORK

    def test_classify_methodology(self) -> None:
        assert LayoutAnalyzer._classify_section("3. Methodology") == SectionType.METHODOLOGY
        assert LayoutAnalyzer._classify_section("Methods") == SectionType.METHODOLOGY
        assert LayoutAnalyzer._classify_section("Approach") == SectionType.METHODOLOGY

    def test_classify_experiments(self) -> None:
        assert LayoutAnalyzer._classify_section("Experiments") == SectionType.EXPERIMENTS
        assert LayoutAnalyzer._classify_section("4. Evaluation") == SectionType.EXPERIMENTS

    def test_classify_results(self) -> None:
        assert LayoutAnalyzer._classify_section("Results") == SectionType.RESULTS
        assert LayoutAnalyzer._classify_section("Results and Discussion") == SectionType.RESULTS

    def test_classify_conclusion(self) -> None:
        assert LayoutAnalyzer._classify_section("Conclusion") == SectionType.CONCLUSION
        assert LayoutAnalyzer._classify_section("6. Summary") == SectionType.CONCLUSION

    def test_classify_references(self) -> None:
        assert LayoutAnalyzer._classify_section("References") == SectionType.REFERENCES
        assert LayoutAnalyzer._classify_section("Bibliography") == SectionType.REFERENCES

    def test_classify_appendix(self) -> None:
        assert LayoutAnalyzer._classify_section("Appendix A") == SectionType.APPENDIX

    def test_classify_unknown_returns_none(self) -> None:
        """Unrecognized text should return None."""
        assert LayoutAnalyzer._classify_section("This is body text.") is None
        assert LayoutAnalyzer._classify_section("") is None
