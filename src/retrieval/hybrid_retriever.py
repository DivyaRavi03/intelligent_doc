"""Hybrid retrieval combining dense vector search and sparse BM25.

The :class:`HybridRetriever` fuses results from :class:`VectorStore`
(semantic similarity) and :class:`BM25Index` (keyword matching) using
Reciprocal Rank Fusion (RRF).  The ``alpha`` parameter controls the
balance: higher alpha favours dense retrieval, lower alpha favours sparse.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.models.schemas import SectionType
from src.retrieval.bm25_index import BM25Index
from src.retrieval.embedding_service import EmbeddingService
from src.retrieval.vector_store import SearchResult, VectorStore

logger = logging.getLogger(__name__)

_DEFAULT_ALPHA = 0.7  # Weight for dense retrieval (1.0 = all dense)
_DEFAULT_CANDIDATES = 20  # Candidates to retrieve from each source
_RRF_K = 60  # Smoothing constant for RRF


@dataclass
class RankedResult:
    """A retrieval result enriched with fusion and reranking scores."""

    chunk_id: str
    text: str
    paper_id: str
    section_type: str
    section_title: str | None = None
    page_numbers: list[int] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    # Scores from pipeline stages
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0


class HybridRetriever:
    """Hybrid dense+sparse retriever with Reciprocal Rank Fusion.

    Args:
        vector_store: Dense vector search backend.
        embedding_service: Service to embed queries for dense retrieval.
        bm25_index: Sparse BM25 keyword index.
        default_alpha: Default weight for dense retrieval
            (``0.0`` = all sparse, ``1.0`` = all dense).
        candidates_per_source: Number of candidates to retrieve from
            each source before fusion.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        bm25_index: BM25Index,
        default_alpha: float = _DEFAULT_ALPHA,
        candidates_per_source: int = _DEFAULT_CANDIDATES,
    ) -> None:
        self._vector_store = vector_store
        self._embedding_service = embedding_service
        self._bm25_index = bm25_index
        self._default_alpha = default_alpha
        self._candidates_per_source = candidates_per_source

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        alpha: float | None = None,
        paper_id: str | None = None,
        section_type: SectionType | None = None,
    ) -> list[RankedResult]:
        """Retrieve and fuse results from dense and sparse sources.

        Args:
            query: The search query.
            top_k: Number of final results to return.
            alpha: Override for the dense/sparse weight (``0.0``–``1.0``).
                Uses *default_alpha* when ``None``.
            paper_id: Optional filter — restrict to a specific paper.
            section_type: Optional filter — restrict to a section type.

        Returns:
            Fused, ranked list of :class:`RankedResult` objects.
        """
        effective_alpha = alpha if alpha is not None else self._default_alpha
        n = self._candidates_per_source

        dense_results = self._get_dense_results(query, n, paper_id, section_type)
        sparse_results = self._get_sparse_results(query, n, paper_id, section_type)

        if not dense_results and not sparse_results:
            return []

        fused = self._reciprocal_rank_fusion(
            dense_results, sparse_results, effective_alpha
        )

        return fused[:top_k]

    # ------------------------------------------------------------------
    # Source retrieval
    # ------------------------------------------------------------------

    def _get_dense_results(
        self,
        query: str,
        n: int,
        paper_id: str | None,
        section_type: SectionType | None,
    ) -> list[SearchResult]:
        """Embed the query and search the vector store."""
        try:
            embedding = self._embedding_service.embed_query(query)
            return self._vector_store.search(
                embedding, n_results=n, paper_id=paper_id, section_type=section_type
            )
        except Exception:
            logger.warning("Dense retrieval failed, falling back to sparse-only", exc_info=True)
            return []

    def _get_sparse_results(
        self,
        query: str,
        n: int,
        paper_id: str | None,
        section_type: SectionType | None,
    ) -> list[SearchResult]:
        """Search the BM25 index."""
        try:
            if not self._bm25_index.is_built():
                logger.warning("BM25 index not built, sparse retrieval skipped")
                return []
            return self._bm25_index.search(
                query, n_results=n, paper_id=paper_id, section_type=section_type
            )
        except Exception:
            logger.warning("Sparse retrieval failed, falling back to dense-only", exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Reciprocal Rank Fusion
    # ------------------------------------------------------------------

    def _reciprocal_rank_fusion(
        self,
        dense_results: list[SearchResult],
        sparse_results: list[SearchResult],
        alpha: float,
    ) -> list[RankedResult]:
        """Fuse two ranked lists using Reciprocal Rank Fusion.

        ``RRF_score(d) = alpha * 1/(k + rank_dense(d))
                        + (1 - alpha) * 1/(k + rank_sparse(d))``

        Chunks missing from one list receive ``rank = 1000`` (penalty).

        Args:
            dense_results: Ranked results from vector search.
            sparse_results: Ranked results from BM25.
            alpha: Weight for the dense component.

        Returns:
            Fused results sorted by RRF score descending.
        """
        k = _RRF_K
        missing_rank = 1000

        # Build rank maps (1-based ranking)
        dense_rank: dict[str, int] = {
            r.chunk_id: rank + 1 for rank, r in enumerate(dense_results)
        }
        sparse_rank: dict[str, int] = {
            r.chunk_id: rank + 1 for rank, r in enumerate(sparse_results)
        }

        # Build lookup for all unique chunks (first occurrence wins)
        all_chunks: dict[str, SearchResult] = {}
        dense_scores: dict[str, float] = {}
        sparse_scores: dict[str, float] = {}

        for r in dense_results:
            if r.chunk_id not in all_chunks:
                all_chunks[r.chunk_id] = r
            dense_scores[r.chunk_id] = r.score

        for r in sparse_results:
            if r.chunk_id not in all_chunks:
                all_chunks[r.chunk_id] = r
            sparse_scores[r.chunk_id] = r.score

        # Compute RRF score for each unique chunk
        fused: list[RankedResult] = []
        for chunk_id, result in all_chunks.items():
            rd = dense_rank.get(chunk_id, missing_rank)
            rs = sparse_rank.get(chunk_id, missing_rank)

            rrf = alpha * (1.0 / (k + rd)) + (1.0 - alpha) * (1.0 / (k + rs))

            fused.append(
                RankedResult(
                    chunk_id=result.chunk_id,
                    text=result.text,
                    paper_id=result.paper_id,
                    section_type=result.section_type,
                    section_title=result.section_title,
                    page_numbers=list(result.page_numbers),
                    metadata=dict(result.metadata),
                    dense_score=dense_scores.get(chunk_id, 0.0),
                    sparse_score=sparse_scores.get(chunk_id, 0.0),
                    rrf_score=round(rrf, 6),
                    rerank_score=0.0,
                    final_score=round(rrf, 6),
                )
            )

        fused.sort(key=lambda r: r.rrf_score, reverse=True)
        return fused
