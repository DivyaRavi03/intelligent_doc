"""Tests for retrieval evaluation metrics and evaluator."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.evaluation.retrieval_eval import (
    RetrievalEvaluator,
    RetrievalEvalResult,
)
from src.retrieval.hybrid_retriever import RankedResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ranked(chunk_id: str, score: float = 0.5) -> RankedResult:
    """Create a minimal RankedResult."""
    return RankedResult(
        chunk_id=chunk_id,
        text=f"text for {chunk_id}",
        paper_id="paper-001",
        section_type="introduction",
        final_score=score,
    )


# ---------------------------------------------------------------------------
# Metric computation tests
# ---------------------------------------------------------------------------


class TestPrecisionAtK:
    """Tests for _precision_at_k."""

    def test_all_relevant(self) -> None:
        retrieved = ["c1", "c2", "c3"]
        relevant = {"c1", "c2", "c3"}
        assert RetrievalEvaluator._precision_at_k(retrieved, relevant, 3) == 1.0

    def test_none_relevant(self) -> None:
        retrieved = ["c1", "c2", "c3"]
        relevant = {"c4", "c5"}
        assert RetrievalEvaluator._precision_at_k(retrieved, relevant, 3) == 0.0

    def test_partial_relevant(self) -> None:
        retrieved = ["c1", "c2", "c3", "c4"]
        relevant = {"c1", "c3"}
        assert RetrievalEvaluator._precision_at_k(retrieved, relevant, 4) == 0.5

    def test_k_larger_than_retrieved(self) -> None:
        retrieved = ["c1", "c2"]
        relevant = {"c1", "c2"}
        assert RetrievalEvaluator._precision_at_k(retrieved, relevant, 5) == 1.0

    def test_empty_retrieved(self) -> None:
        assert RetrievalEvaluator._precision_at_k([], {"c1"}, 3) == 0.0


class TestRecallAtK:
    """Tests for _recall_at_k."""

    def test_all_found(self) -> None:
        retrieved = ["c1", "c2", "c3"]
        relevant = {"c1", "c2"}
        assert RetrievalEvaluator._recall_at_k(retrieved, relevant, 3) == 1.0

    def test_partial_found(self) -> None:
        retrieved = ["c1", "c4"]
        relevant = {"c1", "c2", "c3"}
        result = RetrievalEvaluator._recall_at_k(retrieved, relevant, 2)
        assert abs(result - 1 / 3) < 1e-6

    def test_empty_relevant(self) -> None:
        assert RetrievalEvaluator._recall_at_k(["c1"], set(), 1) == 0.0

    def test_none_found(self) -> None:
        retrieved = ["c4", "c5"]
        relevant = {"c1", "c2"}
        assert RetrievalEvaluator._recall_at_k(retrieved, relevant, 2) == 0.0


class TestReciprocalRank:
    """Tests for _reciprocal_rank."""

    def test_first_hit(self) -> None:
        assert RetrievalEvaluator._reciprocal_rank(["c1", "c2"], {"c1"}) == 1.0

    def test_second_hit(self) -> None:
        assert RetrievalEvaluator._reciprocal_rank(["c1", "c2"], {"c2"}) == 0.5

    def test_third_hit(self) -> None:
        result = RetrievalEvaluator._reciprocal_rank(["c1", "c2", "c3"], {"c3"})
        assert abs(result - 1 / 3) < 1e-6

    def test_no_hit(self) -> None:
        assert RetrievalEvaluator._reciprocal_rank(["c1", "c2"], {"c5"}) == 0.0

    def test_empty_retrieved(self) -> None:
        assert RetrievalEvaluator._reciprocal_rank([], {"c1"}) == 0.0


class TestNDCGAtK:
    """Tests for _ndcg_at_k."""

    def test_perfect_ranking(self) -> None:
        retrieved = ["c1", "c2"]
        relevant = {"c1", "c2"}
        result = RetrievalEvaluator._ndcg_at_k(retrieved, relevant, 2)
        assert abs(result - 1.0) < 1e-6

    def test_empty_relevant(self) -> None:
        assert RetrievalEvaluator._ndcg_at_k(["c1"], set(), 1) == 0.0

    def test_imperfect_ranking(self) -> None:
        # Relevant doc at position 2 instead of 1
        retrieved = ["c2", "c1"]
        relevant = {"c1"}
        result = RetrievalEvaluator._ndcg_at_k(retrieved, relevant, 2)
        assert 0.0 < result < 1.0

    def test_no_relevant_found(self) -> None:
        retrieved = ["c3", "c4"]
        relevant = {"c1", "c2"}
        assert RetrievalEvaluator._ndcg_at_k(retrieved, relevant, 2) == 0.0


# ---------------------------------------------------------------------------
# Evaluator tests
# ---------------------------------------------------------------------------


class TestEvaluate:
    """Tests for evaluate() with mocked retriever."""

    def test_returns_result(self) -> None:
        retriever = MagicMock()
        retriever.retrieve.return_value = [
            _ranked("c1", 0.9),
            _ranked("c2", 0.8),
        ]

        test_set = [
            {
                "query": "test query",
                "paper_id": "paper-001",
                "relevant_chunk_ids": ["c1", "c2"],
                "expected_answer": "answer",
            }
        ]

        evaluator = RetrievalEvaluator()
        result = evaluator.evaluate(test_set, retriever)

        assert isinstance(result, RetrievalEvalResult)
        assert result.num_queries == 1
        assert result.precision_at_k[1] > 0
        assert result.mrr == 1.0

    def test_empty_test_set(self) -> None:
        retriever = MagicMock()
        evaluator = RetrievalEvaluator()
        result = evaluator.evaluate([], retriever)
        assert result.num_queries == 0
        assert result.mrr == 0.0

    def test_perfect_retrieval_scores_1(self) -> None:
        retriever = MagicMock()
        retriever.retrieve.return_value = [_ranked("c1"), _ranked("c2")]

        test_set = [
            {
                "query": "q",
                "paper_id": "p1",
                "relevant_chunk_ids": ["c1", "c2"],
                "expected_answer": "a",
            }
        ]

        evaluator = RetrievalEvaluator(k_values=[1, 3])
        result = evaluator.evaluate(test_set, retriever)
        assert result.precision_at_k[1] == 1.0
        assert result.recall_at_k[3] == 1.0
        assert result.mrr == 1.0

    def test_retriever_failure_handled(self) -> None:
        retriever = MagicMock()
        retriever.retrieve.side_effect = RuntimeError("broken")

        test_set = [
            {
                "query": "q",
                "paper_id": "p1",
                "relevant_chunk_ids": ["c1"],
                "expected_answer": "a",
            }
        ]

        evaluator = RetrievalEvaluator()
        result = evaluator.evaluate(test_set, retriever)
        assert result.num_queries == 1
        assert result.mrr == 0.0

    def test_to_dict_serialisable(self) -> None:
        retriever = MagicMock()
        retriever.retrieve.return_value = [_ranked("c1")]

        test_set = [
            {
                "query": "q",
                "paper_id": "p1",
                "relevant_chunk_ids": ["c1"],
                "expected_answer": "a",
            }
        ]

        evaluator = RetrievalEvaluator()
        result = evaluator.evaluate(test_set, retriever)
        d = result.to_dict()
        assert "precision_at_k" in d
        assert "mrr" in d


class TestLoadTestSet:
    """Tests for load_test_set."""

    def test_loads_fixture(self, tmp_path) -> None:
        import json

        data = [{"query": "test", "relevant_chunk_ids": ["c1"]}]
        path = tmp_path / "test.json"
        path.write_text(json.dumps(data))

        loaded = RetrievalEvaluator.load_test_set(path)
        assert len(loaded) == 1
        assert loaded[0]["query"] == "test"
