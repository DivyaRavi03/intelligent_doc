"""Multi-level paper summarization using map-reduce.

The :class:`PaperSummarizer` produces summaries at three levels of detail:
``one_line`` (≤30 words), ``abstract`` (100-150 words), and ``detailed``
(300-500 words via map-reduce over individual sections).
"""

from __future__ import annotations

import json
import logging
import re

from src.llm.gemini_client import GeminiClient
from src.llm.prompts import (
    SUMMARIZE_ABSTRACT,
    SUMMARIZE_ONE_LINE,
    SUMMARIZE_SECTION,
    SUMMARIZE_SYNTHESIZE,
)
from src.models.schemas import (
    DetectedSection,
    SectionType,
    SummaryLevel,
    SummaryResult,
)

logger = logging.getLogger(__name__)

_ONE_LINE_MAX_WORDS = 30
_ABSTRACT_MAX_WORDS = 150
_DETAILED_MAX_WORDS = 500
_TEXT_TRUNCATE = 8000  # Max chars sent in a single prompt


class PaperSummarizer:
    """Summarize research papers at multiple levels of detail.

    Three levels:

    - **one_line**: max 30 words
    - **abstract**: 100-150 words
    - **detailed**: 300-500 words using map-reduce over sections

    Args:
        client: :class:`GeminiClient` instance.
    """

    def __init__(self, client: GeminiClient | None = None) -> None:
        self._client = client or GeminiClient()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def summarize(
        self,
        paper_id: str,
        text: str,
        sections: list[DetectedSection] | None = None,
        level: str = "abstract",
    ) -> SummaryResult:
        """Summarize a paper at the given level.

        Args:
            paper_id: Identifier for the paper.
            text: Full paper text (used for one_line and abstract levels).
            sections: Parsed sections (required for detailed level).
            level: One of ``"one_line"``, ``"abstract"``, ``"detailed"``.

        Returns:
            :class:`SummaryResult`.

        Raises:
            ValueError: If *level* is not a valid :class:`SummaryLevel`.
        """
        try:
            summary_level = SummaryLevel(level)
        except ValueError:
            raise ValueError(
                f"Invalid summary level '{level}'. "
                f"Must be one of: {', '.join(l.value for l in SummaryLevel)}"
            ) from None

        if summary_level == SummaryLevel.ONE_LINE:
            summary = self._summarize_one_line(text)
            return SummaryResult(
                paper_id=paper_id,
                level=summary_level,
                summary=summary,
                word_count=self._count_words(summary),
                sections_used=[],
            )

        if summary_level == SummaryLevel.ABSTRACT:
            summary = self._summarize_abstract(text)
            return SummaryResult(
                paper_id=paper_id,
                level=summary_level,
                summary=summary,
                word_count=self._count_words(summary),
                sections_used=[],
            )

        # Detailed — requires sections; fall back to abstract if unavailable
        if not sections:
            summary = self._summarize_abstract(text)
            return SummaryResult(
                paper_id=paper_id,
                level=summary_level,
                summary=summary,
                word_count=self._count_words(summary),
                sections_used=[],
            )

        summary, sections_used = self._summarize_detailed(sections)
        return SummaryResult(
            paper_id=paper_id,
            level=summary_level,
            summary=summary,
            word_count=self._count_words(summary),
            sections_used=sections_used,
        )

    # ------------------------------------------------------------------
    # Level-specific strategies
    # ------------------------------------------------------------------

    def _summarize_one_line(self, text: str) -> str:
        """Generate a one-line summary (max 30 words)."""
        try:
            response = self._client.generate(
                SUMMARIZE_ONE_LINE.format(text=text[:_TEXT_TRUNCATE])
            )
            summary = self._extract_summary(response.content)
            # Truncate to max words
            words = summary.split()
            if len(words) > _ONE_LINE_MAX_WORDS:
                summary = " ".join(words[:_ONE_LINE_MAX_WORDS])
            return summary
        except RuntimeError:
            logger.warning("One-line summarization failed")
            return ""

    def _summarize_abstract(self, text: str) -> str:
        """Generate an abstract-length summary (100-150 words)."""
        try:
            response = self._client.generate(
                SUMMARIZE_ABSTRACT.format(text=text[:_TEXT_TRUNCATE])
            )
            return self._extract_summary(response.content)
        except RuntimeError:
            logger.warning("Abstract summarization failed")
            return ""

    def _summarize_detailed(
        self,
        sections: list[DetectedSection],
    ) -> tuple[str, list[str]]:
        """Map-reduce summarization over sections.

        Map: summarize each section independently.
        Reduce: synthesize section summaries into coherent narrative.

        Returns:
            ``(summary, sections_used)``
        """
        # Filter out sections that do not contribute to the summary
        skip = {SectionType.REFERENCES, SectionType.TITLE, SectionType.APPENDIX}
        usable = [s for s in sections if s.section_type not in skip and s.text.strip()]

        if not usable:
            return "", []

        # Map — summarize each section
        section_summaries: list[str] = []
        sections_used: list[str] = []
        for section in usable:
            summary = self._summarize_section(section)
            if summary:
                label = section.title or section.section_type.value
                section_summaries.append(f"{label}: {summary}")
                sections_used.append(section.section_type.value)

        if not section_summaries:
            return "", []

        # Reduce — synthesize
        final = self._synthesize(section_summaries)
        return final, sections_used

    def _summarize_section(self, section: DetectedSection) -> str:
        """Summarize a single section (map step)."""
        try:
            response = self._client.generate(
                SUMMARIZE_SECTION.format(
                    section_title=section.title or "",
                    section_type=section.section_type.value,
                    text=section.text[:_TEXT_TRUNCATE],
                )
            )
            return self._extract_summary(response.content)
        except RuntimeError:
            logger.warning(
                "Section summarization failed for %s",
                section.section_type.value,
            )
            return ""

    def _synthesize(self, section_summaries: list[str]) -> str:
        """Synthesize section summaries into a single narrative (reduce step)."""
        numbered = "\n".join(
            f"{i}. {s}" for i, s in enumerate(section_summaries, 1)
        )
        try:
            response = self._client.generate(
                SUMMARIZE_SYNTHESIZE.format(section_summaries=numbered)
            )
            return self._extract_summary(response.content)
        except RuntimeError:
            logger.warning("Synthesis failed, returning concatenated summaries")
            return " ".join(section_summaries)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_summary(response_text: str) -> str:
        """Extract summary text from a JSON response."""
        try:
            cleaned = PaperSummarizer._strip_markdown_fences(response_text)
            data = json.loads(cleaned)
            if isinstance(data, dict) and "summary" in data:
                return data["summary"]
            return response_text.strip()
        except (json.JSONDecodeError, TypeError):
            return response_text.strip()

    @staticmethod
    def _count_words(text: str) -> int:
        """Count words in text."""
        return len(text.split()) if text else 0

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """Remove markdown code fences."""
        pattern = r"```(?:json)?\s*(.*?)\s*```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1)
        return text.strip()
