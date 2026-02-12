"""Query processing pipeline for intelligent retrieval.

The :class:`QueryProcessor` orchestrates the full retrieval flow: query
classification, parameter tuning, query expansion, hybrid retrieval,
and reranking.  It uses Gemini for classification and expansion, and
delegates retrieval and reranking to :class:`HybridRetriever` and
:class:`GeminiReranker`.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum

from src.config import settings
from src.models.schemas import SectionType
from src.retrieval.hybrid_retriever import HybridRetriever, RankedResult
from src.retrieval.reranker import GeminiReranker

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BASE_BACKOFF = 1.0
_MAX_EXPANSIONS = 3
_RRF_K = 60  # Smoothing constant for cross-query RRF merge


class QueryType(str, Enum):
    """Classification of query intent."""

    FACTUAL = "factual"
    CONCEPTUAL = "conceptual"
    COMPARISON = "comparison"
    METADATA = "metadata"


# Default retrieval parameters per query type
_QUERY_TYPE_PARAMS: dict[QueryType, dict] = {
    QueryType.FACTUAL: {"alpha": 0.5, "top_k": 5},
    QueryType.CONCEPTUAL: {"alpha": 0.8, "top_k": 10},
    QueryType.COMPARISON: {"alpha": 0.7, "top_k": 15},
    QueryType.METADATA: {"alpha": 0.5, "top_k": 5},
}


@dataclass
class QueryResult:
    """Final output of the query processing pipeline."""

    query: str
    query_type: QueryType
    results: list[RankedResult] = field(default_factory=list)
    expanded_queries: list[str] = field(default_factory=list)
    metadata_answer: str | None = None


class QueryProcessor:
    """Full query processing pipeline.

    Pipeline: classify → set params → expand → retrieve → rerank.

    Args:
        hybrid_retriever: The hybrid dense+sparse retriever.
        reranker: The Gemini-based reranker.
        model_name: Gemini model for classification and expansion.
            Defaults to ``settings.gemini_model``.
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: GeminiReranker,
        model_name: str | None = None,
    ) -> None:
        self._retriever = hybrid_retriever
        self._reranker = reranker
        self._model_name = model_name or settings.gemini_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        query: str,
        paper_id: str | None = None,
        section_type: SectionType | None = None,
    ) -> QueryResult:
        """Process a query through the full retrieval pipeline.

        Args:
            query: The user's search query.
            paper_id: Optional paper filter.
            section_type: Optional section filter.

        Returns:
            :class:`QueryResult` with classified type and ranked results.
        """
        if not query or not query.strip():
            return QueryResult(query=query, query_type=QueryType.FACTUAL)

        # 1. Classify
        query_type = self.classify_query(query)
        params = _QUERY_TYPE_PARAMS[query_type]

        # 2. Short-circuit metadata queries
        if query_type == QueryType.METADATA:
            return self._handle_metadata_query(query, paper_id)

        # 3. Expand
        expanded = self.expand_query(query)

        # 4. Retrieve and merge
        candidates = self._retrieve_and_merge(
            queries=expanded,
            top_k=params["top_k"],
            alpha=params["alpha"],
            paper_id=paper_id,
            section_type=section_type,
        )

        # 5. Rerank
        reranked = self._reranker.rerank(query, candidates, top_k=params["top_k"])

        return QueryResult(
            query=query,
            query_type=query_type,
            results=reranked,
            expanded_queries=expanded,
        )

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify_query(self, query: str) -> QueryType:
        """Classify the query intent using Gemini.

        Falls back to ``FACTUAL`` if classification fails.

        Args:
            query: The search query.

        Returns:
            Classified :class:`QueryType`.
        """
        if not settings.gemini_api_key:
            return QueryType.FACTUAL

        prompt = (
            "Classify the following search query into exactly one category.\n\n"
            "Categories:\n"
            "- factual: asking for a specific fact, number, definition, or named entity\n"
            "- conceptual: asking to understand a concept, method, theory, or process\n"
            "- comparison: comparing two or more methods, models, or approaches\n"
            "- metadata: asking about authors, publication date, title, DOI, or journal\n\n"
            f"Query: {query}\n\n"
            'Return ONLY valid JSON: {{"type": "category_name"}}'
        )

        try:
            response_text = self._call_gemini(prompt)
            cleaned = self._strip_markdown_fences(response_text)
            data = json.loads(cleaned)
            type_str = data.get("type", "factual").lower().strip()
            return QueryType(type_str)
        except (json.JSONDecodeError, ValueError, RuntimeError) as exc:
            logger.warning("Query classification failed (%s), defaulting to FACTUAL", exc)
            return QueryType.FACTUAL

    # ------------------------------------------------------------------
    # Query expansion
    # ------------------------------------------------------------------

    def expand_query(self, query: str) -> list[str]:
        """Generate 2–3 query reformulations using Gemini.

        Always includes the original query. Falls back to ``[query]`` on
        failure.

        Args:
            query: The original search query.

        Returns:
            List of queries (original + reformulations).
        """
        if not settings.gemini_api_key:
            return [query]

        prompt = (
            "Generate 2-3 alternative phrasings of the following search query "
            "that would help retrieve relevant passages from a research paper. "
            "Each reformulation should emphasise different aspects or use "
            "different terminology.\n\n"
            f"Original query: {query}\n\n"
            'Return ONLY valid JSON: {{"queries": ["reformulation1", "reformulation2"]}}'
        )

        try:
            response_text = self._call_gemini(prompt)
            cleaned = self._strip_markdown_fences(response_text)
            data = json.loads(cleaned)

            queries = data.get("queries", [])
            if not isinstance(queries, list):
                return [query]

            # Ensure strings and limit count
            expansions = [str(q) for q in queries if q][:_MAX_EXPANSIONS]
            return [query] + expansions

        except (json.JSONDecodeError, ValueError, RuntimeError) as exc:
            logger.warning("Query expansion failed (%s), using original query", exc)
            return [query]

    # ------------------------------------------------------------------
    # Retrieval and merge
    # ------------------------------------------------------------------

    def _retrieve_and_merge(
        self,
        queries: list[str],
        top_k: int,
        alpha: float,
        paper_id: str | None,
        section_type: SectionType | None,
    ) -> list[RankedResult]:
        """Retrieve for each query and merge results via cross-query RRF.

        Args:
            queries: List of queries (original + expansions).
            top_k: Number of results per query retrieval.
            alpha: Dense/sparse balance.
            paper_id: Optional paper filter.
            section_type: Optional section filter.

        Returns:
            Merged and deduplicated results.
        """
        if not queries:
            return []

        # Single query — no merge needed
        if len(queries) == 1:
            return self._retriever.retrieve(
                queries[0],
                top_k=top_k,
                alpha=alpha,
                paper_id=paper_id,
                section_type=section_type,
            )

        # Multiple queries — retrieve for each and merge via RRF
        per_query_results: list[list[RankedResult]] = []
        for q in queries:
            try:
                results = self._retriever.retrieve(
                    q,
                    top_k=top_k,
                    alpha=alpha,
                    paper_id=paper_id,
                    section_type=section_type,
                )
                per_query_results.append(results)
            except Exception:
                logger.warning("Retrieval failed for expanded query: %s", q, exc_info=True)
                per_query_results.append([])

        return self._cross_query_rrf(per_query_results)

    def _cross_query_rrf(
        self, per_query_results: list[list[RankedResult]]
    ) -> list[RankedResult]:
        """Merge results from multiple queries via Reciprocal Rank Fusion.

        Chunks appearing in multiple queries' results get boosted.
        """
        k = _RRF_K
        missing_rank = 1000

        # Build rank maps per query (1-based)
        rank_maps: list[dict[str, int]] = []
        for results in per_query_results:
            rank_map = {r.chunk_id: i + 1 for i, r in enumerate(results)}
            rank_maps.append(rank_map)

        # Collect all unique chunks (first occurrence wins)
        all_chunks: dict[str, RankedResult] = {}
        for results in per_query_results:
            for r in results:
                if r.chunk_id not in all_chunks:
                    all_chunks[r.chunk_id] = r

        # Compute cross-query RRF score
        merged: list[RankedResult] = []
        num_queries = len(per_query_results)
        for chunk_id, result in all_chunks.items():
            rrf_sum = 0.0
            for rank_map in rank_maps:
                rank = rank_map.get(chunk_id, missing_rank)
                rrf_sum += 1.0 / (k + rank)
            rrf_score = rrf_sum / num_queries

            merged.append(
                RankedResult(
                    chunk_id=result.chunk_id,
                    text=result.text,
                    paper_id=result.paper_id,
                    section_type=result.section_type,
                    section_title=result.section_title,
                    page_numbers=list(result.page_numbers),
                    metadata=dict(result.metadata),
                    dense_score=result.dense_score,
                    sparse_score=result.sparse_score,
                    rrf_score=round(rrf_score, 6),
                    rerank_score=0.0,
                    final_score=round(rrf_score, 6),
                )
            )

        merged.sort(key=lambda r: r.rrf_score, reverse=True)
        return merged

    # ------------------------------------------------------------------
    # Metadata short-circuit
    # ------------------------------------------------------------------

    def _handle_metadata_query(
        self, query: str, paper_id: str | None
    ) -> QueryResult:
        """Handle metadata queries without retrieval."""
        if paper_id:
            answer = (
                f"This is a metadata query about paper '{paper_id}'. "
                "Use the document metadata endpoint for author, title, DOI, "
                "and publication information."
            )
        else:
            answer = (
                "This is a metadata query. Please specify a paper ID to "
                "retrieve metadata such as authors, title, DOI, and "
                "publication date."
            )

        return QueryResult(
            query=query,
            query_type=QueryType.METADATA,
            results=[],
            expanded_queries=[query],
            metadata_answer=answer,
        )

    # ------------------------------------------------------------------
    # Gemini API
    # ------------------------------------------------------------------

    def _call_gemini(self, prompt: str) -> str:
        """Call Gemini with exponential-backoff retry.

        Args:
            prompt: The prompt to send.

        Returns:
            Raw response text.

        Raises:
            RuntimeError: If all retries are exhausted.
        """
        import google.generativeai as genai

        genai.configure(api_key=settings.gemini_api_key)
        model = genai.GenerativeModel(self._model_name)

        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json",
                        temperature=0.0,
                    ),
                )
                return response.text
            except Exception as exc:
                last_exc = exc
                wait = _BASE_BACKOFF * (2**attempt)
                logger.warning(
                    "Gemini call attempt %d/%d failed (%s), retrying in %.1fs",
                    attempt + 1,
                    _MAX_RETRIES,
                    exc,
                    wait,
                )
                time.sleep(wait)

        raise RuntimeError(
            f"Gemini call failed after {_MAX_RETRIES} retries: {last_exc}"
        ) from last_exc

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """Remove markdown code fences from text."""
        pattern = r"```(?:json)?\s*(.*?)\s*```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1)
        return text.strip()
