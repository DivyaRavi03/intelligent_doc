"""Tests for QA evaluation using LLM-as-Judge."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from src.evaluation.qa_eval import QAEvaluator, QAEvalResult
from src.models.schemas import (
    Citation,
    ClaimVerification,
    LLMResponse,
    QAResponse,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _llm_response(content: str) -> LLMResponse:
    """Create a minimal LLMResponse."""
    return LLMResponse(
        content=content,
        input_tokens=10,
        output_tokens=5,
        latency_ms=100.0,
        model="gemini-2.0-flash",
        cached=False,
        cost_usd=0.0001,
    )


# ---------------------------------------------------------------------------
# Judge tests
# ---------------------------------------------------------------------------


class TestJudgeAnswer:
    """Tests for _judge_answer()."""

    def test_returns_scores(self) -> None:
        judge_client = MagicMock()
        judge_json = json.dumps(
            {
                "faithfulness": 5,
                "relevance": 4,
                "completeness": 3,
                "explanation": "Good answer",
            }
        )
        judge_client.generate.return_value = _llm_response(judge_json)

        evaluator = QAEvaluator(judge_client=judge_client)
        f, r, c, exp = evaluator._judge_answer("query", "answer", "context")

        assert f == 5.0
        assert r == 4.0
        assert c == 3.0
        assert exp == "Good answer"

    def test_failure_returns_zeros(self) -> None:
        judge_client = MagicMock()
        judge_client.generate.side_effect = RuntimeError("API error")

        evaluator = QAEvaluator(judge_client=judge_client)
        f, r, c, exp = evaluator._judge_answer("query", "answer", "context")

        assert f == 0.0
        assert r == 0.0
        assert c == 0.0

    def test_invalid_json_returns_zeros(self) -> None:
        judge_client = MagicMock()
        judge_client.generate.return_value = _llm_response("not json at all")

        evaluator = QAEvaluator(judge_client=judge_client)
        f, r, c, exp = evaluator._judge_answer("query", "answer", "context")

        assert f == 0.0
        assert r == 0.0
        assert c == 0.0

    def test_markdown_fences_stripped(self) -> None:
        judge_client = MagicMock()
        fenced = '```json\n{"faithfulness":4,"relevance":3,"completeness":2,"explanation":"ok"}\n```'
        judge_client.generate.return_value = _llm_response(fenced)

        evaluator = QAEvaluator(judge_client=judge_client)
        f, r, c, exp = evaluator._judge_answer("q", "a", "ctx")

        assert f == 4.0
        assert r == 3.0
        assert c == 2.0


# ---------------------------------------------------------------------------
# Evaluator tests
# ---------------------------------------------------------------------------


class TestEvaluate:
    """Tests for evaluate() with mocked QA engine and judge."""

    def test_returns_result(self) -> None:
        judge_client = MagicMock()
        judge_json = json.dumps(
            {
                "faithfulness": 4,
                "relevance": 5,
                "completeness": 4,
                "explanation": "OK",
            }
        )
        judge_client.generate.return_value = _llm_response(judge_json)

        qa_engine = MagicMock()
        qa_engine.answer.return_value = QAResponse(
            query="test query",
            answer="The answer is X [1].",
            citations=[
                Citation(
                    source_index=1,
                    chunk_id="c1",
                    text_snippet="text",
                    paper_id="p1",
                    section_type="introduction",
                )
            ],
            claim_verifications=[],
            faithfulness_score=0.9,
        )

        test_set = [
            {
                "query": "test query",
                "paper_id": "paper-001",
                "relevant_chunk_ids": ["c1"],
                "expected_answer": "The answer is X.",
            }
        ]

        evaluator = QAEvaluator(judge_client=judge_client)
        result = evaluator.evaluate(test_set, qa_engine)

        assert isinstance(result, QAEvalResult)
        assert result.num_queries == 1
        assert result.avg_faithfulness == 4.0
        assert result.answer_rate == 1.0
        assert result.citation_rate == 1.0
        assert result.hallucination_rate == 0.0

    def test_empty_test_set(self) -> None:
        judge_client = MagicMock()
        qa_engine = MagicMock()

        evaluator = QAEvaluator(judge_client=judge_client)
        result = evaluator.evaluate([], qa_engine)

        assert result.num_queries == 0
        assert result.avg_faithfulness == 0.0

    def test_no_answer_detected(self) -> None:
        judge_client = MagicMock()
        judge_json = json.dumps(
            {"faithfulness": 1, "relevance": 1, "completeness": 1, "explanation": ""}
        )
        judge_client.generate.return_value = _llm_response(judge_json)

        qa_engine = MagicMock()
        qa_engine.answer.return_value = QAResponse(
            query="q",
            answer="Unable to generate an answer.",
            citations=[],
            claim_verifications=[],
            faithfulness_score=0.0,
        )

        test_set = [
            {
                "query": "q",
                "paper_id": "p1",
                "relevant_chunk_ids": [],
                "expected_answer": "expected",
            }
        ]

        evaluator = QAEvaluator(judge_client=judge_client)
        result = evaluator.evaluate(test_set, qa_engine)

        assert result.answer_rate == 0.0
        assert result.hallucination_rate == 1.0  # faithfulness_score < 0.5

    def test_qa_engine_failure_handled(self) -> None:
        judge_client = MagicMock()
        qa_engine = MagicMock()
        qa_engine.answer.side_effect = RuntimeError("broken")

        test_set = [
            {
                "query": "q",
                "paper_id": "p1",
                "relevant_chunk_ids": [],
                "expected_answer": "expected",
            }
        ]

        evaluator = QAEvaluator(judge_client=judge_client)
        result = evaluator.evaluate(test_set, qa_engine)

        assert result.num_queries == 1
        assert result.avg_faithfulness == 0.0

    def test_to_dict(self) -> None:
        result = QAEvalResult(
            num_queries=5,
            avg_faithfulness=4.0,
            avg_relevance=3.5,
            avg_completeness=4.2,
            answer_rate=0.8,
            citation_rate=0.6,
            hallucination_rate=0.1,
        )
        d = result.to_dict()
        assert d["num_queries"] == 5
        assert d["avg_faithfulness"] == 4.0
        assert d["hallucination_rate"] == 0.1
