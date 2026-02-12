"""Metadata extraction from research papers.

Combines fast heuristic extraction (regex patterns on the first few pages)
with an optional LLM pass (Gemini JSON mode) for higher accuracy. The two
results are merged, preferring the LLM output when both are available.
"""

from __future__ import annotations

import json
import logging
import re

from src.config import settings
from src.models.schemas import DocumentMetadataSchema, PageResult

logger = logging.getLogger(__name__)

# Maximum characters from the start of the paper sent to the LLM
_LLM_CONTEXT_CHARS = 4000
# Number of leading pages used for heuristic extraction
_HEURISTIC_PAGES = 3


class MetadataExtractor:
    """Extract bibliographic metadata from research paper text.

    Two strategies are combined:
    1. **Heuristic** — regex patterns applied to the first few pages.
    2. **LLM** — Gemini JSON-mode extraction for structured fields.

    The final output merges both, favouring LLM results for overlapping fields.
    """

    def extract(
        self, pages: list[PageResult], *, use_llm: bool = True
    ) -> DocumentMetadataSchema:
        """Extract metadata from page text.

        Args:
            pages: Extracted page results (at least the first 2-3 pages).
            use_llm: Whether to attempt LLM extraction. Set to ``False``
                in tests or when no API key is available.

        Returns:
            Merged :class:`DocumentMetadataSchema`.
        """
        if not pages:
            return DocumentMetadataSchema()

        heuristic = self._heuristic_extract(pages)

        if use_llm and settings.gemini_api_key:
            llm = self._llm_extract(pages)
            return self._merge(heuristic, llm)

        return heuristic

    # ------------------------------------------------------------------
    # Heuristic extraction
    # ------------------------------------------------------------------

    def _heuristic_extract(self, pages: list[PageResult]) -> DocumentMetadataSchema:
        """Regex-based metadata extraction from the first pages."""
        text = "\n".join(p.text for p in pages[:_HEURISTIC_PAGES])
        lines = [l.strip() for l in text.split("\n") if l.strip()]

        title = self._extract_title(lines)
        authors = self._extract_authors(lines, title)
        abstract = self._extract_abstract(text)
        doi = self._extract_doi(text)
        keywords = self._extract_keywords(text)

        return DocumentMetadataSchema(
            title=title,
            authors=authors,
            abstract=abstract,
            doi=doi,
            keywords=keywords,
            confidence=0.5,
        )

    @staticmethod
    def _extract_title(lines: list[str]) -> str | None:
        """Heuristic: the title is usually the first long, non-boilerplate line."""
        skip_patterns = re.compile(
            r"(?i)^(proceedings|journal|volume|issue|copyright|arxiv|preprint|doi\s*:)"
        )
        for line in lines[:15]:
            if skip_patterns.match(line):
                continue
            if len(line) < 10:
                continue
            # Title lines are typically 20-300 chars
            if 10 <= len(line) <= 300:
                return line
        return None

    @staticmethod
    def _extract_authors(lines: list[str], title: str | None) -> list[str]:
        """Heuristic: authors usually appear right after the title."""
        if not title:
            return []

        # Find title position
        title_idx = None
        for i, line in enumerate(lines):
            if line == title:
                title_idx = i
                break

        if title_idx is None:
            return []

        authors: list[str] = []
        # Check the next few lines after the title
        for line in lines[title_idx + 1: title_idx + 5]:
            # Skip lines that look like affiliations/emails
            if re.search(r"@|university|department|institute|lab\b", line, re.IGNORECASE):
                continue
            # Skip numbered/keyword lines
            if re.match(r"^\d+\s|^abstract|^keyword", line, re.IGNORECASE):
                break
            # Authors are often comma or "and" separated
            if re.search(r"[A-Z][a-z]+", line):
                # Split on common separators
                parts = re.split(r"\s*[,;]\s*|\s+and\s+", line)
                for part in parts:
                    part = part.strip()
                    # Keep only parts that look like names (2+ words, starting with uppercase)
                    if part and re.match(r"^[A-Z][a-z]+([\s-][A-Z][a-z]+)*$", part):
                        authors.append(part)
                if authors:
                    break

        return authors

    @staticmethod
    def _extract_abstract(text: str) -> str | None:
        """Extract the abstract section."""
        # Look for explicit "Abstract" heading
        match = re.search(
            r"(?i)\babstract\b[:\s\-]*\n?(.*?)(?=\n\s*\n\s*(?:\d+\.?\s+)?[A-Z][a-z]|\n\s*(?:keywords?|introduction)\b)",
            text,
            re.DOTALL,
        )
        if match:
            abstract = match.group(1).strip()
            # Collapse internal whitespace
            abstract = re.sub(r"\s+", " ", abstract)
            if len(abstract) > 50:
                return abstract

        return None

    @staticmethod
    def _extract_doi(text: str) -> str | None:
        """Extract a DOI from the text."""
        match = re.search(r"(?i)\b(?:doi\s*:?\s*)(10\.\d{4,}/[^\s]+)", text)
        if match:
            return match.group(1).rstrip(".")
        # Also look for bare DOIs
        match = re.search(r"\b(10\.\d{4,}/\S+)", text)
        if match:
            return match.group(1).rstrip(".")
        return None

    @staticmethod
    def _extract_keywords(text: str) -> list[str]:
        """Extract keywords from a 'Keywords:' line."""
        match = re.search(r"(?i)\bkeywords?\s*[:\-]\s*(.+?)(?:\n\n|\n\s*\d)", text, re.DOTALL)
        if match:
            raw = match.group(1).strip()
            # Split on comma, semicolon, or bullet
            parts = re.split(r"\s*[;,•·]\s*", raw)
            return [p.strip() for p in parts if p.strip() and len(p.strip()) > 1]
        return []

    # ------------------------------------------------------------------
    # LLM extraction
    # ------------------------------------------------------------------

    def _llm_extract(self, pages: list[PageResult]) -> DocumentMetadataSchema:
        """Use Gemini JSON mode to extract structured metadata."""
        context = "\n".join(p.text for p in pages[:_HEURISTIC_PAGES])[:_LLM_CONTEXT_CHARS]

        try:
            import google.generativeai as genai
        except ImportError:
            logger.warning("google-generativeai not installed; skipping LLM extraction")
            return DocumentMetadataSchema()

        genai.configure(api_key=settings.gemini_api_key)
        model = genai.GenerativeModel(settings.gemini_model)

        prompt = (
            "You are a metadata extraction system for academic research papers.\n"
            "Given the following text from the beginning of a paper, extract:\n"
            '- "title": the paper title\n'
            '- "authors": list of author full names\n'
            '- "abstract": the abstract text\n'
            '- "doi": the DOI if present\n'
            '- "journal": the journal or conference name if present\n'
            '- "publication_date": publication date/year if present\n'
            '- "keywords": list of keywords if present\n\n'
            "Return ONLY valid JSON. If a field is not found, use null for strings "
            "and empty arrays for lists.\n\n"
            f"---\n{context}\n---"
        )

        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.0,
                ),
            )
            data = json.loads(response.text)
        except Exception:
            logger.error("Gemini metadata extraction failed", exc_info=True)
            return DocumentMetadataSchema()

        return DocumentMetadataSchema(
            title=data.get("title"),
            authors=data.get("authors", []),
            abstract=data.get("abstract"),
            doi=data.get("doi"),
            journal=data.get("journal"),
            publication_date=data.get("publication_date"),
            keywords=data.get("keywords", []),
            confidence=0.85,
        )

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    @staticmethod
    def _merge(
        heuristic: DocumentMetadataSchema,
        llm: DocumentMetadataSchema,
    ) -> DocumentMetadataSchema:
        """Merge heuristic and LLM results, preferring LLM for non-empty fields."""
        return DocumentMetadataSchema(
            title=llm.title or heuristic.title,
            authors=llm.authors if llm.authors else heuristic.authors,
            abstract=llm.abstract or heuristic.abstract,
            doi=llm.doi or heuristic.doi,
            journal=llm.journal or heuristic.journal,
            publication_date=llm.publication_date or heuristic.publication_date,
            keywords=llm.keywords if llm.keywords else heuristic.keywords,
            references_count=heuristic.references_count,
            confidence=max(heuristic.confidence, llm.confidence),
        )
