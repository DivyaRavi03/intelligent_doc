"""BM25 sparse keyword index for document chunks.

The :class:`BM25Index` builds an in-memory BM25Okapi index over
:class:`EnrichedChunk` objects, enabling keyword-based retrieval that
complements dense vector search.  BM25 excels at exact keyword matching
for technical terms, author names, and methodology-specific vocabulary
where embedding models may falter.
"""

from __future__ import annotations

import logging
import string

from src.models.schemas import EnrichedChunk, SectionType
from src.retrieval.vector_store import SearchResult

logger = logging.getLogger(__name__)

# Minimal English stopwords — avoids an external dependency (nltk, spacy).
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "because", "but", "and", "or", "if", "while", "about",
    "that", "this", "these", "those", "it", "its", "i", "we", "they",
    "he", "she", "what", "which", "who", "whom",
})

# Pre-built translation table to strip punctuation
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


class BM25Index:
    """In-memory BM25Okapi index over enriched chunks.

    Args:
        remove_stopwords: Whether to filter English stopwords during
            tokenisation.  Defaults to ``True``.
    """

    def __init__(self, *, remove_stopwords: bool = True) -> None:
        self._remove_stopwords = remove_stopwords
        self._chunks: list[EnrichedChunk] = []
        self._bm25 = None  # rank_bm25.BM25Okapi (lazy)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_index(self, chunks: list[EnrichedChunk]) -> None:
        """Build the BM25 index from enriched chunks.

        Args:
            chunks: The document chunks to index.

        Raises:
            ValueError: If *chunks* is empty.
        """
        if not chunks:
            raise ValueError("Cannot build BM25 index from empty chunk list")

        from rank_bm25 import BM25Okapi

        self._chunks = list(chunks)
        tokenized_corpus = [self._tokenize(c.text) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)
        logger.info("BM25 index built with %d chunks", len(self._chunks))

    def search(
        self,
        query: str,
        n_results: int = 10,
        paper_id: str | None = None,
        section_type: SectionType | None = None,
    ) -> list[SearchResult]:
        """Search the BM25 index.

        Args:
            query: The search query text.
            n_results: Maximum number of results.
            paper_id: Optional filter — restrict to a specific paper.
            section_type: Optional filter — restrict to a section type.

        Returns:
            Ranked list of :class:`SearchResult` objects with BM25 scores
            normalised to the ``0.0``–``1.0`` range.

        Raises:
            RuntimeError: If the index has not been built yet.
        """
        if self._bm25 is None:
            raise RuntimeError(
                "BM25 index has not been built. Call build_index() first."
            )

        tokens = self._tokenize(query)
        if not tokens:
            return []

        raw_scores = self._bm25.get_scores(tokens)

        # Pair each chunk index with its score
        scored: list[tuple[int, float]] = [
            (idx, float(score))
            for idx, score in enumerate(raw_scores)
            if score > 0.0
        ]

        # Apply metadata filters
        scored = self._filter_by_metadata(scored, paper_id, section_type)

        if not scored:
            return []

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # Normalise scores to [0, 1]
        max_score = scored[0][1]
        if max_score <= 0.0:
            return []

        # Truncate to n_results
        scored = scored[:n_results]

        results: list[SearchResult] = []
        for idx, score in scored:
            chunk = self._chunks[idx]
            page_str = ",".join(str(p) for p in chunk.page_numbers)
            results.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=round(score / max_score, 4),
                    paper_id=chunk.paper_id,
                    section_type=chunk.section_type.value,
                    section_title=chunk.section_title,
                    page_numbers=list(chunk.page_numbers),
                    metadata={
                        "paper_title": chunk.paper_title or "",
                        "chunk_index": chunk.chunk_index,
                        "token_count": chunk.token_count,
                    },
                )
            )

        return results

    def is_built(self) -> bool:
        """Return ``True`` if the index has been built."""
        return self._bm25 is not None

    def chunk_count(self) -> int:
        """Return the number of indexed chunks."""
        return len(self._chunks)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> list[str]:
        """Tokenise text for BM25.

        Lowercase → remove punctuation → split whitespace → optional
        stopword removal.

        Args:
            text: Raw text to tokenise.

        Returns:
            List of tokens.
        """
        lowered = text.lower()
        cleaned = lowered.translate(_PUNCT_TABLE)
        tokens = cleaned.split()

        if self._remove_stopwords:
            tokens = [t for t in tokens if t not in _STOPWORDS]

        return tokens

    def _filter_by_metadata(
        self,
        scored: list[tuple[int, float]],
        paper_id: str | None,
        section_type: SectionType | None,
    ) -> list[tuple[int, float]]:
        """Filter scored results by optional metadata criteria."""
        if not paper_id and not section_type:
            return scored

        filtered: list[tuple[int, float]] = []
        for idx, score in scored:
            chunk = self._chunks[idx]
            if paper_id and chunk.paper_id != paper_id:
                continue
            if section_type and chunk.section_type != section_type:
                continue
            filtered.append((idx, score))

        return filtered
