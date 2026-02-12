"""Chunk-size optimisation via grid search.

The :class:`ChunkOptimizer` evaluates different ``(chunk_size, overlap)``
configurations against a set of test queries with known-relevant sections.
It measures **Precision@5** — of the top-5 retrieved chunks, how many come
from the correct section — and returns the configuration that maximises it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.chunking.chunker import SectionAwareChunker
from src.models.schemas import EnrichedChunk, PaperStructure, SectionType

logger = logging.getLogger(__name__)

_DEFAULT_CHUNK_SIZES = [256, 512, 768, 1024]
_DEFAULT_OVERLAPS = [0, 25, 50, 100]


@dataclass
class TestQuery:
    """A test query with known-relevant section types for evaluation."""

    query: str
    relevant_section_types: list[SectionType]


@dataclass
class OptimizationResult:
    """Result of a grid-search optimisation run."""

    best_chunk_size: int
    best_overlap: int
    best_precision: float
    all_results: list[dict] = field(default_factory=list)


class ChunkOptimizer:
    """Find optimal chunk size and overlap via grid search.

    Evaluates each ``(chunk_size, overlap)`` pair by:
    1. Chunking the paper with those parameters.
    2. For each test query, scoring retrieved chunks against the known-
       relevant section types.
    3. Averaging Precision@5 across all queries.

    The "retrieval" step is simulated using simple term-overlap scoring
    (no embedding model needed), making optimisation fast and free.

    Args:
        chunk_sizes: List of target token counts to try.
        overlaps: List of overlap token counts to try.
    """

    def __init__(
        self,
        chunk_sizes: list[int] | None = None,
        overlaps: list[int] | None = None,
    ) -> None:
        self.chunk_sizes = chunk_sizes or list(_DEFAULT_CHUNK_SIZES)
        self.overlaps = overlaps or list(_DEFAULT_OVERLAPS)

    def optimize(
        self,
        paper: PaperStructure,
        test_queries: list[TestQuery],
    ) -> OptimizationResult:
        """Run grid search and return the best configuration.

        Args:
            paper: A fully-parsed paper to chunk with each configuration.
            test_queries: Queries with known-relevant section types.

        Returns:
            :class:`OptimizationResult` with the winning parameters.
        """
        if not test_queries:
            # No queries to evaluate — return sensible defaults
            return OptimizationResult(
                best_chunk_size=512,
                best_overlap=50,
                best_precision=0.0,
            )

        best_precision = -1.0
        best_size = self.chunk_sizes[0]
        best_overlap = self.overlaps[0]
        all_results: list[dict] = []

        for size in self.chunk_sizes:
            for overlap in self.overlaps:
                if overlap >= size:
                    continue  # overlap must be smaller than chunk size

                chunker = SectionAwareChunker(
                    target_tokens=size, overlap_tokens=overlap
                )
                chunks = chunker.chunk(paper)

                if not chunks:
                    continue

                avg_p5 = self._evaluate(chunks, test_queries)
                entry = {
                    "chunk_size": size,
                    "overlap": overlap,
                    "num_chunks": len(chunks),
                    "precision_at_5": round(avg_p5, 4),
                }
                all_results.append(entry)
                logger.debug("Grid search: size=%d overlap=%d p@5=%.4f", size, overlap, avg_p5)

                if avg_p5 > best_precision:
                    best_precision = avg_p5
                    best_size = size
                    best_overlap = overlap

        return OptimizationResult(
            best_chunk_size=best_size,
            best_overlap=best_overlap,
            best_precision=round(best_precision, 4),
            all_results=all_results,
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        chunks: list[EnrichedChunk],
        queries: list[TestQuery],
    ) -> float:
        """Average Precision@5 across all test queries."""
        total_p5 = 0.0

        for query in queries:
            scored = self._score_chunks(chunks, query.query)
            top5 = scored[:5]
            relevant = sum(
                1
                for chunk in top5
                if chunk.section_type in query.relevant_section_types
            )
            total_p5 += relevant / min(5, len(top5)) if top5 else 0.0

        return total_p5 / len(queries)

    @staticmethod
    def _score_chunks(
        chunks: list[EnrichedChunk], query: str
    ) -> list[EnrichedChunk]:
        """Rank chunks by simple term-overlap relevance to *query*.

        This is a lightweight proxy for embedding-based retrieval,
        suitable for offline optimisation without API calls.
        """
        query_terms = set(query.lower().split())

        scored: list[tuple[float, int, EnrichedChunk]] = []
        for idx, chunk in enumerate(chunks):
            chunk_terms = set(chunk.text.lower().split())
            if not query_terms:
                overlap = 0.0
            else:
                overlap = len(query_terms & chunk_terms) / len(query_terms)
            # Secondary sort by chunk_index for stability
            scored.append((overlap, -idx, chunk))

        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [c for _, _, c in scored]
