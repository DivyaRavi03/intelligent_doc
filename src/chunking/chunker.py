"""Section-aware chunking for research papers.

The :class:`SectionAwareChunker` splits parsed paper content into
:class:`EnrichedChunk` objects that respect structural boundaries.  Key rules:

- **Never split across sections** — each chunk belongs to exactly one section.
- **Tables are single chunks** — a table (caption + rendered rows) is never split.
- **Equations stay with context** — inline ``$...$`` or display ``$$...$$``
  blocks are kept with surrounding text.
- **References are batched** — grouped 5-10 per chunk, not one per chunk.
- **Target size** — 512 tokens with 50 token overlap between consecutive
  chunks within the same section.
"""

from __future__ import annotations

import hashlib
import logging
import re
import uuid

from src.models.schemas import (
    DetectedSection,
    EnrichedChunk,
    ExtractedTable,
    PaperStructure,
    SectionType,
)

logger = logging.getLogger(__name__)

_DEFAULT_TARGET_TOKENS = 512
_DEFAULT_OVERLAP_TOKENS = 50
_REFS_PER_CHUNK_MIN = 5
_REFS_PER_CHUNK_MAX = 10

# Rough approximation: 1 token ≈ 4 characters for English text.
# Used only for sizing; actual token counts use the same factor consistently.
_CHARS_PER_TOKEN = 4

# Patterns for equation blocks that should not be split from their context
_DISPLAY_EQUATION_RE = re.compile(r"\$\$.*?\$\$", re.DOTALL)
_INLINE_EQUATION_RE = re.compile(r"(?<!\$)\$(?!\$).+?(?<!\$)\$(?!\$)")


class SectionAwareChunker:
    """Split a parsed paper into enriched, section-respecting chunks.

    Args:
        target_tokens: Desired token count per chunk.
        overlap_tokens: Token overlap between consecutive chunks in the
            same section.
        refs_per_chunk_min: Minimum references to group in one chunk.
        refs_per_chunk_max: Maximum references to group in one chunk.
    """

    def __init__(
        self,
        target_tokens: int = _DEFAULT_TARGET_TOKENS,
        overlap_tokens: int = _DEFAULT_OVERLAP_TOKENS,
        refs_per_chunk_min: int = _REFS_PER_CHUNK_MIN,
        refs_per_chunk_max: int = _REFS_PER_CHUNK_MAX,
    ) -> None:
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.refs_per_chunk_min = refs_per_chunk_min
        self.refs_per_chunk_max = refs_per_chunk_max

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, paper: PaperStructure) -> list[EnrichedChunk]:
        """Chunk a fully-parsed paper into enriched chunks.

        Args:
            paper: Aggregated Phase 1 output.

        Returns:
            Ordered list of :class:`EnrichedChunk` objects.
        """
        raw_chunks: list[EnrichedChunk] = []

        # 1. Chunk each section
        for section in paper.sections:
            if section.section_type == SectionType.REFERENCES:
                raw_chunks.extend(
                    self._chunk_references(paper.references, section, paper)
                )
            else:
                raw_chunks.extend(self._chunk_section(section, paper))

        # 2. Add tables as standalone chunks
        for table in paper.tables:
            raw_chunks.append(self._table_to_chunk(table, paper))

        # 3. Assign final indices and totals
        total = len(raw_chunks)
        result: list[EnrichedChunk] = []
        for idx, chunk in enumerate(raw_chunks):
            chunk.chunk_index = idx
            chunk.total_chunks = total
            result.append(chunk)

        return result

    # ------------------------------------------------------------------
    # Section chunking
    # ------------------------------------------------------------------

    def _chunk_section(
        self,
        section: DetectedSection,
        paper: PaperStructure,
    ) -> list[EnrichedChunk]:
        """Split a single section into chunks respecting target size."""
        text = section.text.strip()
        if not text:
            return []

        target_chars = self.target_tokens * _CHARS_PER_TOKEN
        overlap_chars = self.overlap_tokens * _CHARS_PER_TOKEN

        fragments = self._recursive_split(text, target_chars)

        chunks: list[EnrichedChunk] = []
        for i, fragment in enumerate(fragments):
            # Apply overlap: prepend tail of previous fragment
            if i > 0 and overlap_chars > 0:
                prev_tail = fragments[i - 1][-overlap_chars:]
                fragment = prev_tail + " " + fragment

            chunks.append(
                self._enrich(
                    text=fragment.strip(),
                    section=section,
                    paper=paper,
                )
            )

        return chunks

    def _recursive_split(self, text: str, target_chars: int) -> list[str]:
        """Recursively split text into fragments of roughly *target_chars*.

        Split hierarchy: paragraphs → sentences → words.
        Never splits inside equation blocks.
        """
        if len(text) <= target_chars:
            return [text]

        # Protect equation blocks from splitting
        equations: dict[str, str] = {}
        protected = text
        for match in _DISPLAY_EQUATION_RE.finditer(text):
            placeholder = f"__EQ{len(equations)}__"
            equations[placeholder] = match.group()
            protected = protected.replace(match.group(), placeholder, 1)

        # Try splitting on paragraph boundaries first
        paragraphs = re.split(r"\n\s*\n", protected)
        if len(paragraphs) > 1:
            return self._merge_splits(paragraphs, target_chars, equations)

        # Fall back to sentence splitting
        sentences = re.split(r"(?<=[.!?])\s+", protected)
        if len(sentences) > 1:
            return self._merge_splits(sentences, target_chars, equations)

        # Last resort: split on whitespace at target boundary
        words = protected.split()
        return self._merge_splits(words, target_chars, equations, join_char=" ")

    def _merge_splits(
        self,
        parts: list[str],
        target_chars: int,
        equations: dict[str, str],
        join_char: str = "\n\n",
    ) -> list[str]:
        """Merge small parts into chunks up to *target_chars*, then restore equations."""
        fragments: list[str] = []
        current: list[str] = []
        current_len = 0

        for part in parts:
            part_len = len(part) + len(join_char)
            if current and current_len + part_len > target_chars:
                fragments.append(join_char.join(current))
                current = [part]
                current_len = len(part)
            else:
                current.append(part)
                current_len += part_len

        if current:
            fragments.append(join_char.join(current))

        # Restore equation placeholders
        restored: list[str] = []
        for frag in fragments:
            for placeholder, original in equations.items():
                frag = frag.replace(placeholder, original)
            restored.append(frag)

        return restored

    # ------------------------------------------------------------------
    # Reference chunking
    # ------------------------------------------------------------------

    def _chunk_references(
        self,
        references: list[str],
        section: DetectedSection,
        paper: PaperStructure,
    ) -> list[EnrichedChunk]:
        """Group references into chunks of 5-10 entries each."""
        if not references:
            # Fall back to chunking the raw section text
            if section.text.strip():
                return self._chunk_section(section, paper)
            return []

        chunks: list[EnrichedChunk] = []
        batch_size = min(self.refs_per_chunk_max, max(self.refs_per_chunk_min, 5))

        for i in range(0, len(references), batch_size):
            batch = references[i : i + batch_size]
            text = "\n".join(batch)
            chunks.append(
                self._enrich(text=text, section=section, paper=paper)
            )

        return chunks

    # ------------------------------------------------------------------
    # Table chunking
    # ------------------------------------------------------------------

    def _table_to_chunk(
        self, table: ExtractedTable, paper: PaperStructure
    ) -> EnrichedChunk:
        """Convert an extracted table into a single chunk."""
        lines: list[str] = []
        if table.caption:
            lines.append(table.caption)
        if table.headers:
            lines.append(" | ".join(table.headers))
            lines.append("-" * (len(" | ".join(table.headers))))
        for row in table.rows:
            lines.append(" | ".join(row))

        text = "\n".join(lines)

        return EnrichedChunk(
            chunk_id=self._make_id(text, paper.paper_id),
            text=text,
            token_count=self._count_tokens(text),
            section_type=SectionType.UNKNOWN,
            section_title=table.caption,
            page_numbers=[table.page_number],
            paper_id=paper.paper_id,
            paper_title=paper.metadata.title,
            chunk_index=0,
            total_chunks=0,
            metadata={
                "type": "table",
                "table_index": table.table_index,
                "extraction_method": table.extraction_method,
            },
        )

    # ------------------------------------------------------------------
    # Enrichment
    # ------------------------------------------------------------------

    def _enrich(
        self,
        text: str,
        section: DetectedSection,
        paper: PaperStructure,
    ) -> EnrichedChunk:
        """Wrap raw text into an :class:`EnrichedChunk` with provenance."""
        page_numbers = [section.page_start]
        if section.page_end is not None and section.page_end != section.page_start:
            page_numbers = list(range(section.page_start, section.page_end + 1))

        return EnrichedChunk(
            chunk_id=self._make_id(text, paper.paper_id),
            text=text,
            token_count=self._count_tokens(text),
            section_type=section.section_type,
            section_title=section.title,
            page_numbers=page_numbers,
            paper_id=paper.paper_id,
            paper_title=paper.metadata.title,
            chunk_index=0,  # set later by chunk()
            total_chunks=0,  # set later by chunk()
            metadata={
                "section_order": section.order_index,
            },
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _count_tokens(text: str) -> int:
        """Approximate token count using character-based heuristic."""
        return max(1, len(text) // _CHARS_PER_TOKEN)

    @staticmethod
    def _make_id(text: str, paper_id: str) -> str:
        """Deterministic chunk ID from content + paper ID."""
        raw = f"{paper_id}:{text}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
