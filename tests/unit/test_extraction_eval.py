"""Tests for extraction evaluation metrics and evaluator."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from src.evaluation.extraction_eval import (
    ExtractionEvaluator,
    ExtractionEvalResult,
)
from src.models.schemas import LLMResponse


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
# Metric helper tests
# ---------------------------------------------------------------------------


class TestExactMatch:
    """Tests for _exact_match."""

    def test_identical(self) -> None:
        assert ExtractionEvaluator._exact_match("Hello World", "hello world") == 1.0

    def test_different(self) -> None:
        assert ExtractionEvaluator._exact_match("Hello", "World") == 0.0

    def test_with_whitespace(self) -> None:
        assert ExtractionEvaluator._exact_match("  hello  ", "hello") == 1.0

    def test_empty(self) -> None:
        assert ExtractionEvaluator._exact_match("", "") == 1.0


class TestExactMatchList:
    """Tests for _exact_match_list."""

    def test_same_set(self) -> None:
        assert ExtractionEvaluator._exact_match_list(
            ["Alice", "Bob"], ["bob", "alice"]
        ) == 1.0

    def test_different_set(self) -> None:
        assert ExtractionEvaluator._exact_match_list(
            ["Alice"], ["Bob"]
        ) == 0.0

    def test_subset(self) -> None:
        assert ExtractionEvaluator._exact_match_list(
            ["Alice"], ["Alice", "Bob"]
        ) == 0.0


class TestRougeL:
    """Tests for _rouge_l."""

    def test_identical(self) -> None:
        score = ExtractionEvaluator._rouge_l(
            "the cat sat on the mat", "the cat sat on the mat"
        )
        assert score > 0.99

    def test_partial_overlap(self) -> None:
        score = ExtractionEvaluator._rouge_l(
            "the cat sat", "the cat sat on the mat"
        )
        assert 0.0 < score < 1.0

    def test_no_overlap(self) -> None:
        score = ExtractionEvaluator._rouge_l("hello world", "foo bar baz")
        assert score < 0.5

    def test_empty_predicted(self) -> None:
        assert ExtractionEvaluator._rouge_l("", "some text") == 0.0


class TestKeywordPrecisionRecall:
    """Tests for _keyword_precision_recall."""

    def test_perfect(self) -> None:
        p, r = ExtractionEvaluator._keyword_precision_recall(
            ["deep learning", "NLP"], ["deep learning", "NLP"]
        )
        assert p == 1.0 and r == 1.0

    def test_partial(self) -> None:
        p, r = ExtractionEvaluator._keyword_precision_recall(
            ["deep learning", "CV"], ["deep learning", "NLP"]
        )
        assert p == 0.5 and r == 0.5

    def test_empty_predicted(self) -> None:
        p, r = ExtractionEvaluator._keyword_precision_recall(
            [], ["deep learning"]
        )
        assert p == 0.0 and r == 0.0

    def test_no_overlap(self) -> None:
        p, r = ExtractionEvaluator._keyword_precision_recall(
            ["a", "b"], ["c", "d"]
        )
        assert p == 0.0 and r == 0.0


# ---------------------------------------------------------------------------
# Evaluator tests
# ---------------------------------------------------------------------------


class TestEvaluate:
    """Tests for evaluate() with mocked extractor."""

    def test_returns_result(self) -> None:
        client = MagicMock()
        findings_json = json.dumps(
            {
                "findings": [
                    {
                        "claim": "novel approach to document understanding",
                        "supporting_quote": "This paper presents a novel approach",
                        "confidence": 0.9,
                    }
                ]
            }
        )
        meth_json = json.dumps(
            {"approach": "transformer", "datasets": [], "tools": [], "eval_metrics": []}
        )
        results_json = json.dumps({"results": []})
        client.generate.side_effect = [
            _llm_response(findings_json),
            _llm_response(findings_json),  # dual-prompt
            _llm_response(meth_json),
            _llm_response(results_json),
        ]

        from src.llm.extractor import PaperExtractor

        extractor = PaperExtractor(client=client)

        test_set = [
            {
                "paper_id": "test-paper-001",
                "paper_text": (
                    "Test Title\nAuthor One\n\nAbstract\nTest abstract.\n\n"
                    "Keywords: ml, ai\n\n1. Introduction\nSome text."
                ),
                "expected_title": "Test Title",
                "expected_authors": ["Author One"],
                "expected_abstract": "Test abstract.",
                "expected_keywords": ["ml", "ai"],
                "expected_findings": [
                    "novel approach to document understanding"
                ],
            }
        ]

        evaluator = ExtractionEvaluator()
        result = evaluator.evaluate(test_set, extractor)

        assert isinstance(result, ExtractionEvalResult)
        assert result.num_papers == 1
        assert 0.0 <= result.overall_accuracy <= 1.0
        assert "title" in result.per_field_accuracy
        assert "findings" in result.per_field_accuracy

    def test_empty_test_set(self) -> None:
        extractor = MagicMock()
        evaluator = ExtractionEvaluator()
        result = evaluator.evaluate([], extractor)
        assert result.num_papers == 0
        assert result.overall_accuracy == 0.0

    def test_to_dict(self) -> None:
        result = ExtractionEvalResult(
            num_papers=1,
            per_field_accuracy={"title": 1.0},
            overall_accuracy=0.8,
        )
        d = result.to_dict()
        assert d["num_papers"] == 1
        assert d["overall_accuracy"] == 0.8
