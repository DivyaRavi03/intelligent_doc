"""Unit tests for multi-level paper summarization."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from src.llm.summarizer import PaperSummarizer
from src.models.schemas import (
    DetectedSection,
    LLMResponse,
    SectionType,
    SummaryLevel,
    SummaryResult,
)


def _llm_response(content: str) -> LLMResponse:
    """Helper to create a minimal LLMResponse."""
    return LLMResponse(
        content=content,
        input_tokens=10,
        output_tokens=5,
        latency_ms=100.0,
        model="gemini-2.0-flash",
        cached=False,
        cost_usd=0.0001,
    )


def _section(
    section_type: SectionType,
    text: str,
    title: str | None = None,
) -> DetectedSection:
    """Helper to create a minimal DetectedSection."""
    return DetectedSection(
        section_type=section_type,
        title=title,
        text=text,
        page_start=0,
        page_end=0,
        order_index=0,
    )


_SAMPLE_TEXT = (
    "We propose a novel transformer architecture for document understanding. "
    "Our model achieves 94.2% F1 on the DocBank benchmark, surpassing the "
    "previous state-of-the-art by 3.1 points."
)


class TestOneLine:
    """Tests for one-line summarization."""

    def test_one_line_returns_summary(self) -> None:
        """Should return a SummaryResult with one_line level."""
        client = MagicMock()
        summary_json = json.dumps({"summary": "Novel transformer achieves 94.2% F1 on DocBank."})
        client.generate.return_value = _llm_response(summary_json)

        summarizer = PaperSummarizer(client=client)
        result = summarizer.summarize("paper-001", _SAMPLE_TEXT, level="one_line")

        assert isinstance(result, SummaryResult)
        assert result.level == SummaryLevel.ONE_LINE
        assert result.paper_id == "paper-001"
        assert len(result.summary) > 0

    def test_one_line_truncates_to_30_words(self) -> None:
        """Should truncate summaries exceeding 30 words."""
        client = MagicMock()
        long_summary = " ".join(["word"] * 50)
        summary_json = json.dumps({"summary": long_summary})
        client.generate.return_value = _llm_response(summary_json)

        summarizer = PaperSummarizer(client=client)
        result = summarizer.summarize("paper-001", _SAMPLE_TEXT, level="one_line")

        assert result.word_count <= 30

    def test_one_line_api_failure_returns_empty(self) -> None:
        """LLM failure should return empty summary string."""
        client = MagicMock()
        client.generate.side_effect = RuntimeError("API error")

        summarizer = PaperSummarizer(client=client)
        result = summarizer.summarize("paper-001", _SAMPLE_TEXT, level="one_line")

        assert result.summary == ""
        assert result.word_count == 0


class TestAbstract:
    """Tests for abstract-level summarization."""

    def test_abstract_returns_summary(self) -> None:
        """Should return an abstract-level summary."""
        client = MagicMock()
        summary_json = json.dumps({"summary": "A comprehensive summary of the paper."})
        client.generate.return_value = _llm_response(summary_json)

        summarizer = PaperSummarizer(client=client)
        result = summarizer.summarize("paper-001", _SAMPLE_TEXT, level="abstract")

        assert result.level == SummaryLevel.ABSTRACT
        assert len(result.summary) > 0

    def test_abstract_is_default_level(self) -> None:
        """Default level should be 'abstract'."""
        client = MagicMock()
        summary_json = json.dumps({"summary": "Default summary."})
        client.generate.return_value = _llm_response(summary_json)

        summarizer = PaperSummarizer(client=client)
        result = summarizer.summarize("paper-001", _SAMPLE_TEXT)

        assert result.level == SummaryLevel.ABSTRACT

    def test_abstract_api_failure_returns_empty(self) -> None:
        """LLM failure should return empty summary."""
        client = MagicMock()
        client.generate.side_effect = RuntimeError("API error")

        summarizer = PaperSummarizer(client=client)
        result = summarizer.summarize("paper-001", _SAMPLE_TEXT, level="abstract")

        assert result.summary == ""


class TestDetailed:
    """Tests for detailed map-reduce summarization."""

    def test_detailed_uses_sections(self) -> None:
        """Should summarize each section then synthesize."""
        client = MagicMock()

        section_summary = json.dumps({"summary": "Section summary."})
        final_summary = json.dumps({"summary": "Final synthesized summary of the paper."})

        # 2 sections → 2 map calls + 1 reduce call
        client.generate.side_effect = [
            _llm_response(section_summary),
            _llm_response(section_summary),
            _llm_response(final_summary),
        ]

        sections = [
            _section(SectionType.INTRODUCTION, "Intro text here.", "1. Introduction"),
            _section(SectionType.METHODOLOGY, "Method text here.", "2. Methodology"),
        ]

        summarizer = PaperSummarizer(client=client)
        result = summarizer.summarize(
            "paper-001", _SAMPLE_TEXT, sections=sections, level="detailed"
        )

        assert result.level == SummaryLevel.DETAILED
        assert "Final synthesized summary" in result.summary
        assert len(result.sections_used) == 2

    def test_detailed_skips_references_and_title(self) -> None:
        """Should skip REFERENCES, TITLE, and APPENDIX sections."""
        client = MagicMock()

        section_summary = json.dumps({"summary": "Intro summary."})
        final_summary = json.dumps({"summary": "Only intro."})

        client.generate.side_effect = [
            _llm_response(section_summary),
            _llm_response(final_summary),
        ]

        sections = [
            _section(SectionType.TITLE, "Paper Title"),
            _section(SectionType.INTRODUCTION, "Intro text.", "1. Introduction"),
            _section(SectionType.REFERENCES, "[1] Smith et al."),
            _section(SectionType.APPENDIX, "Extra tables."),
        ]

        summarizer = PaperSummarizer(client=client)
        result = summarizer.summarize(
            "paper-001", _SAMPLE_TEXT, sections=sections, level="detailed"
        )

        assert "introduction" in result.sections_used
        assert "references" not in result.sections_used
        assert "title" not in result.sections_used

    def test_detailed_fallback_when_no_sections(self) -> None:
        """Should fall back to abstract-level when no sections provided."""
        client = MagicMock()
        summary_json = json.dumps({"summary": "Abstract fallback summary."})
        client.generate.return_value = _llm_response(summary_json)

        summarizer = PaperSummarizer(client=client)
        result = summarizer.summarize("paper-001", _SAMPLE_TEXT, sections=None, level="detailed")

        assert result.level == SummaryLevel.DETAILED
        assert result.sections_used == []

    def test_detailed_empty_sections_list(self) -> None:
        """Should fall back to abstract-level with empty sections list."""
        client = MagicMock()
        summary_json = json.dumps({"summary": "Fallback."})
        client.generate.return_value = _llm_response(summary_json)

        summarizer = PaperSummarizer(client=client)
        result = summarizer.summarize("paper-001", _SAMPLE_TEXT, sections=[], level="detailed")

        assert result.sections_used == []

    def test_detailed_section_failure_skips_section(self) -> None:
        """If a section summarization fails, it should be skipped."""
        client = MagicMock()

        # First section fails, second succeeds, then synthesis
        client.generate.side_effect = [
            RuntimeError("API error"),
            _llm_response(json.dumps({"summary": "Method summary."})),
            _llm_response(json.dumps({"summary": "Final from one section."})),
        ]

        sections = [
            _section(SectionType.INTRODUCTION, "Intro text.", "1. Introduction"),
            _section(SectionType.METHODOLOGY, "Method text.", "2. Methodology"),
        ]

        summarizer = PaperSummarizer(client=client)
        result = summarizer.summarize(
            "paper-001", _SAMPLE_TEXT, sections=sections, level="detailed"
        )

        # Only methodology should appear in sections_used
        assert "methodology" in result.sections_used
        assert len(result.sections_used) == 1


class TestLevelValidation:
    """Tests for level validation and edge cases."""

    def test_invalid_level_raises_value_error(self) -> None:
        """Invalid level should raise ValueError."""
        summarizer = PaperSummarizer()
        with pytest.raises(ValueError, match="Invalid summary level"):
            summarizer.summarize("paper-001", _SAMPLE_TEXT, level="invalid")

    def test_word_count_tracked(self) -> None:
        """word_count should match actual word count of summary."""
        client = MagicMock()
        summary_json = json.dumps({"summary": "This is a five word summary."})
        client.generate.return_value = _llm_response(summary_json)

        summarizer = PaperSummarizer(client=client)
        result = summarizer.summarize("paper-001", _SAMPLE_TEXT, level="abstract")

        assert result.word_count == 6  # "This is a five word summary."

    def test_extract_summary_from_json(self) -> None:
        """Should extract 'summary' field from JSON response."""
        result = PaperSummarizer._extract_summary('{"summary": "extracted text"}')
        assert result == "extracted text"

    def test_extract_summary_non_json_returns_stripped(self) -> None:
        """Non-JSON response should be returned stripped."""
        result = PaperSummarizer._extract_summary("  plain text  ")
        assert result == "plain text"

    def test_strip_markdown_fences(self) -> None:
        """Should remove markdown code fences."""
        text = '```json\n{"summary": "test"}\n```'
        result = PaperSummarizer._strip_markdown_fences(text)
        assert result == '{"summary": "test"}'
