"""Retrieval evaluation with precision, recall, MRR, and NDCG metrics.

The :class:`RetrievalEvaluator` runs test queries against a retriever
and computes standard information retrieval metrics at multiple *k* values.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.retrieval.hybrid_retriever import HybridRetriever, RankedResult

logger = logging.getLogger(__name__)

_DEFAULT_K_VALUES = [1, 3, 5, 10]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class QueryEvalResult:
    """Per-query evaluation result."""

    query: str
    paper_id: str
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    relevant_chunk_ids: list[str] = field(default_factory=list)
    precision_at_k: dict[int, float] = field(default_factory=dict)
    recall_at_k: dict[int, float] = field(default_factory=dict)
    reciprocal_rank: float = 0.0
    ndcg_at_k: dict[int, float] = field(default_factory=dict)


@dataclass
class RetrievalEvalResult:
    """Aggregated retrieval evaluation result."""

    num_queries: int = 0
    precision_at_k: dict[int, float] = field(default_factory=dict)
    recall_at_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg_at_k: dict[int, float] = field(default_factory=dict)
    per_query: list[QueryEvalResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dict."""
        return {
            "num_queries": self.num_queries,
            "precision_at_k": {str(k): v for k, v in self.precision_at_k.items()},
            "recall_at_k": {str(k): v for k, v in self.recall_at_k.items()},
            "mrr": self.mrr,
            "ndcg_at_k": {str(k): v for k, v in self.ndcg_at_k.items()},
        }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class RetrievalEvaluator:
    """Evaluate retrieval quality against ground-truth test sets.

    Args:
        k_values: List of *k* values for precision@k, recall@k, NDCG@k.
    """

    def __init__(self, k_values: list[int] | None = None) -> None:
        self._k_values = k_values or list(_DEFAULT_K_VALUES)

    def evaluate(
        self,
        test_set: list[dict],
        retriever: HybridRetriever,
    ) -> RetrievalEvalResult:
        """Run evaluation on test queries.

        Args:
            test_set: List of dicts with keys ``query``, ``paper_id``,
                ``relevant_chunk_ids``, ``expected_answer``.
            retriever: The retriever to evaluate.

        Returns:
            Aggregated metrics and per-query breakdown.
        """
        if not test_set:
            return RetrievalEvalResult()

        per_query: list[QueryEvalResult] = []
        max_k = max(self._k_values)

        for entry in test_set:
            query = entry["query"]
            paper_id = entry.get("paper_id")
            relevant = set(entry.get("relevant_chunk_ids", []))

            try:
                results: list[RankedResult] = retriever.retrieve(
                    query, top_k=max_k, paper_id=paper_id
                )
            except Exception:
                logger.warning("Retrieval failed for query: %s", query)
                results = []

            retrieved = [r.chunk_id for r in results]

            q_result = QueryEvalResult(
                query=query,
                paper_id=paper_id or "",
                retrieved_chunk_ids=retrieved,
                relevant_chunk_ids=list(relevant),
                reciprocal_rank=self._reciprocal_rank(retrieved, relevant),
            )

            for k in self._k_values:
                q_result.precision_at_k[k] = self._precision_at_k(
                    retrieved, relevant, k
                )
                q_result.recall_at_k[k] = self._recall_at_k(retrieved, relevant, k)
                q_result.ndcg_at_k[k] = self._ndcg_at_k(retrieved, relevant, k)

            per_query.append(q_result)

        # Aggregate across queries
        n = len(per_query)
        agg_precision: dict[int, float] = {}
        agg_recall: dict[int, float] = {}
        agg_ndcg: dict[int, float] = {}

        for k in self._k_values:
            agg_precision[k] = sum(q.precision_at_k[k] for q in per_query) / n
            agg_recall[k] = sum(q.recall_at_k[k] for q in per_query) / n
            agg_ndcg[k] = sum(q.ndcg_at_k[k] for q in per_query) / n

        mrr = sum(q.reciprocal_rank for q in per_query) / n

        return RetrievalEvalResult(
            num_queries=n,
            precision_at_k=agg_precision,
            recall_at_k=agg_recall,
            mrr=mrr,
            ndcg_at_k=agg_ndcg,
            per_query=per_query,
        )

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def load_test_set(path: str | Path) -> list[dict]:
        """Load a test set from a JSON file."""
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def _precision_at_k(
        retrieved: list[str], relevant: set[str], k: int
    ) -> float:
        """Precision@k: fraction of top-k results that are relevant."""
        top_k = retrieved[:k]
        if not top_k:
            return 0.0
        return sum(1 for cid in top_k if cid in relevant) / len(top_k)

    @staticmethod
    def _recall_at_k(
        retrieved: list[str], relevant: set[str], k: int
    ) -> float:
        """Recall@k: fraction of all relevant docs found in top-k."""
        if not relevant:
            return 0.0
        top_k = retrieved[:k]
        return sum(1 for cid in top_k if cid in relevant) / len(relevant)

    @staticmethod
    def _reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
        """Reciprocal rank: 1/rank of first relevant result."""
        for i, cid in enumerate(retrieved):
            if cid in relevant:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def _ndcg_at_k(
        retrieved: list[str], relevant: set[str], k: int
    ) -> float:
        """NDCG@k with binary relevance."""
        dcg = 0.0
        for i, cid in enumerate(retrieved[:k]):
            if cid in relevant:
                dcg += 1.0 / math.log2(i + 2)  # +2 because i is 0-indexed

        # Ideal DCG: all relevant docs in top positions
        ideal_count = min(len(relevant), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))
        if idcg == 0.0:
            return 0.0
        return dcg / idcg
