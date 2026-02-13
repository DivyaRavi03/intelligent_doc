"""Structural layout analysis for research papers.

Detects logical sections (Abstract, Introduction, Methodology, …) from the
extracted page texts. Uses heuristic rules based on heading patterns, font
sizes, and common section naming conventions in academic papers.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from src.models.schemas import DetectedSection, LayoutAnalysisResult, PageResult, SectionType

logger = logging.getLogger(__name__)

# Patterns for common research paper section headings
_SECTION_PATTERNS: dict[SectionType, list[re.Pattern[str]]] = {
    SectionType.ABSTRACT: [
        re.compile(r"^\s*abstract\s*$", re.IGNORECASE),
    ],
    SectionType.INTRODUCTION: [
        re.compile(r"^\s*\d*\.?\s*introduction\s*$", re.IGNORECASE),
    ],
    SectionType.RELATED_WORK: [
        re.compile(r"^\s*\d*\.?\s*related\s+work\s*$", re.IGNORECASE),
        re.compile(r"^\s*\d*\.?\s*literature\s+review\s*$", re.IGNORECASE),
        re.compile(r"^\s*\d*\.?\s*background\s*$", re.IGNORECASE),
        re.compile(r"^\s*\d*\.?\s*prior\s+work\s*$", re.IGNORECASE),
    ],
    SectionType.METHODOLOGY: [
        re.compile(r"^\s*\d*\.?\s*method(?:ology|s)?\s*$", re.IGNORECASE),
        re.compile(r"^\s*\d*\.?\s*approach\s*$", re.IGNORECASE),
        re.compile(r"^\s*\d*\.?\s*proposed\s+(?:method|approach|system)\s*$", re.IGNORECASE),
        re.compile(r"^\s*\d*\.?\s*model\s*$", re.IGNORECASE),
    ],
    SectionType.EXPERIMENTS: [
        re.compile(r"^\s*\d*\.?\s*experiments?\s*$", re.IGNORECASE),
        re.compile(r"^\s*\d*\.?\s*experimental\s+(?:setup|results?)\s*$", re.IGNORECASE),
        re.compile(r"^\s*\d*\.?\s*evaluation\s*$", re.IGNORECASE),
    ],
    SectionType.RESULTS: [
        re.compile(r"^\s*\d*\.?\s*results?\s*$", re.IGNORECASE),
        re.compile(r"^\s*\d*\.?\s*results?\s+and\s+(?:discussion|analysis)\s*$", re.IGNORECASE),
    ],
    SectionType.DISCUSSION: [
        re.compile(r"^\s*\d*\.?\s*discussion\s*$", re.IGNORECASE),
        re.compile(r"^\s*\d*\.?\s*analysis\s*$", re.IGNORECASE),
    ],
    SectionType.CONCLUSION: [
        re.compile(r"^\s*\d*\.?\s*conclusions?\s*$", re.IGNORECASE),
        re.compile(r"^\s*\d*\.?\s*(?:conclusion|summary)\s+and\s+future\s+work\s*$", re.IGNORECASE),
        re.compile(r"^\s*\d*\.?\s*summary\s*$", re.IGNORECASE),
    ],
    SectionType.REFERENCES: [
        re.compile(r"^\s*\d*\.?\s*references?\s*$", re.IGNORECASE),
        re.compile(r"^\s*\d*\.?\s*bibliography\s*$", re.IGNORECASE),
    ],
    SectionType.APPENDIX: [
        re.compile(r"^\s*appendix", re.IGNORECASE),
        re.compile(r"^\s*supplementary\s+material", re.IGNORECASE),
    ],
}

# Regex to detect numbered section headings like "1. Introduction" or "IV. METHODS"
_NUMBERED_HEADING_RE = re.compile(
    r"^(?:"
    r"\d+\.?\s+"  # Arabic: "1. " or "1 "
    r"|[IVXLC]+\.?\s+"  # Roman: "IV. " or "IV "
    r")"
    r"[A-Z]",  # Heading text starts with uppercase
)

# Reference line patterns (e.g., "[1] Author, ...")
_REFERENCE_ENTRY_RE = re.compile(
    r"^\s*\[(\d+)\]\s+(.+)",
)


@dataclass
class _SectionAccumulator:
    """Internal mutable container used while building sections."""

    section_type: SectionType
    title: str | None
    lines: list[str] = field(default_factory=list)
    page_start: int = 0
    page_end: int | None = None


class LayoutAnalyzer:
    """Analyse the structural layout of a research paper.

    Takes page-level text extraction results and returns detected logical
    sections (Abstract, Introduction, etc.) with their content.
    """

    def analyze(self, pages: list[PageResult]) -> LayoutAnalysisResult:
        """Run layout analysis over extracted pages.

        Args:
            pages: Ordered list of page extraction results.

        Returns:
            :class:`LayoutAnalysisResult` with sections and references.
        """
        if not pages:
            return LayoutAnalysisResult(sections=[], references=[])

        sections = self._build_sections(pages)
        references = self._extract_references(sections)

        return LayoutAnalysisResult(sections=sections, references=references)

    # ------------------------------------------------------------------
    # Core section detection
    # ------------------------------------------------------------------

    def _build_sections(self, pages: list[PageResult]) -> list[DetectedSection]:
        """Walk through pages line-by-line, splitting at detected headings."""
        accumulators: list[_SectionAccumulator] = []
        current = _SectionAccumulator(
            section_type=SectionType.TITLE,
            title=None,
            page_start=pages[0].page_number,
        )

        for page in pages:
            for line in page.text.split("\n"):
                stripped = line.strip()
                if not stripped:
                    current.lines.append("")
                    continue

                detected = self._classify_section(stripped)
                if detected is not None and self._detect_headers(stripped):
                    # Close the current section
                    current.page_end = page.page_number
                    accumulators.append(current)
                    # Start a new section
                    current = _SectionAccumulator(
                        section_type=detected,
                        title=stripped,
                        page_start=page.page_number,
                    )
                else:
                    current.lines.append(line)

            # Track page range
            current.page_end = page.page_number

        # Don't forget the last section
        accumulators.append(current)

        # Convert to DetectedSection objects
        sections: list[DetectedSection] = []
        for idx, acc in enumerate(accumulators):
            text = "\n".join(acc.lines).strip()
            if not text and acc.section_type == SectionType.TITLE and not acc.title:
                continue  # skip empty leading section
            sections.append(
                DetectedSection(
                    section_type=acc.section_type,
                    title=acc.title,
                    text=text,
                    page_start=acc.page_start,
                    page_end=acc.page_end,
                    order_index=idx,
                )
            )

        return sections

    # ------------------------------------------------------------------
    # Heading heuristics
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_headers(line: str) -> bool:
        """Heuristic: is this line likely a section heading?

        Headings are typically short, may be numbered, and are often
        uppercase or title-case.
        """
        stripped = line.strip()
        if not stripped or len(stripped) > 120:
            return False

        # Numbered headings: "1. Introduction", "III. Methods"
        if _NUMBERED_HEADING_RE.match(stripped):
            return True

        # ALL-CAPS headings
        alpha_chars = [c for c in stripped if c.isalpha()]
        if alpha_chars and all(c.isupper() for c in alpha_chars) and len(alpha_chars) >= 3:
            return True

        # Short title-case lines that match known section patterns
        words = stripped.split()
        if 1 <= len(words) <= 6:
            for patterns in _SECTION_PATTERNS.values():
                for pat in patterns:
                    if pat.match(stripped):
                        return True

        return False

    @staticmethod
    def _classify_section(line: str) -> SectionType | None:
        """Classify a heading line into a :class:`SectionType`.

        Returns ``None`` if the line doesn't match any known section.
        """
        stripped = line.strip()
        for section_type, patterns in _SECTION_PATTERNS.items():
            for pat in patterns:
                if pat.match(stripped):
                    return section_type

        # Fallback: numbered heading that didn't match known types
        if _NUMBERED_HEADING_RE.match(stripped):
            return SectionType.UNKNOWN

        return None

    # ------------------------------------------------------------------
    # Reference extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_references(sections: list[DetectedSection]) -> list[str]:
        """Pull individual reference entries from the References section."""
        references: list[str] = []
        ref_sections = [s for s in sections if s.section_type == SectionType.REFERENCES]
        if not ref_sections:
            return references

        ref_text = ref_sections[0].text
        current_ref: list[str] = []

        for line in ref_text.split("\n"):
            match = _REFERENCE_ENTRY_RE.match(line)
            if match:
                if current_ref:
                    references.append(" ".join(current_ref).strip())
                current_ref = [line.strip()]
            elif current_ref and line.strip():
                current_ref.append(line.strip())

        if current_ref:
            references.append(" ".join(current_ref).strip())

        return references
