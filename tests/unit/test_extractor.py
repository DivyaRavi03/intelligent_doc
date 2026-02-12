"""Unit tests for structured paper extraction with confidence scoring."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.llm.extractor import PaperExtractor
from src.models.schemas import (
    Finding,
    LLMResponse,
    MethodologyExtraction,
    PaperExtraction,
    ResultExtraction,
)


def _llm_response(content: str) -> LLMResponse:
    """Helper to create a minimal LLMResponse."""
    return LLMResponse(
        content=content,
        input_tokens=10,
        output_tokens=5,
        latency_ms=100.0,
        model="gemini-2.0-flash",
        cached=False,
        cost_usd=0.0001,
    )


_SAMPLE_TEXT = (
    "We propose a novel transformer architecture for document understanding. "
    "Our model achieves 94.2% F1 on the DocBank benchmark, surpassing the "
    "previous state-of-the-art by 3.1 points. The approach uses multi-modal "
    "fusion of text and visual features."
)


class TestExtractKeyFindings:
    """Tests for :meth:`PaperExtractor.extract_key_findings`."""

    # ------------------------------------------------------------------
    # Successful extraction
    # ------------------------------------------------------------------

    def test_extract_findings_returns_list(self) -> None:
        """Should return a list of Finding objects."""
        client = MagicMock()
        findings_json = json.dumps({
            "findings": [
                {
                    "claim": "Model achieves 94.2% F1",
                    "supporting_quote": "Our model achieves 94.2% F1",
                    "confidence": 0.9,
                }
            ]
        })
        client.generate.return_value = _llm_response(findings_json)

        extractor = PaperExtractor(client=client)
        results = extractor.extract_key_findings(_SAMPLE_TEXT)

        assert len(results) >= 1
        assert isinstance(results[0], Finding)
        assert results[0].claim == "Model achieves 94.2% F1"

    def test_extract_findings_confidence_bounded(self) -> None:
        """Confidence should be clamped to [0.0, 1.0]."""
        client = MagicMock()
        findings_json = json.dumps({
            "findings": [
                {"claim": "test", "supporting_quote": "test quote", "confidence": 0.9},
            ]
        })
        client.generate.return_value = _llm_response(findings_json)

        extractor = PaperExtractor(client=client)
        results = extractor.extract_key_findings(_SAMPLE_TEXT)

        for f in results:
            assert 0.0 <= f.confidence <= 1.0

    def test_extract_findings_empty_text_returns_empty(self) -> None:
        """Empty text should return empty list without calling LLM."""
        client = MagicMock()
        extractor = PaperExtractor(client=client)
        results = extractor.extract_key_findings("")
        assert results == []
        client.generate.assert_not_called()

    def test_extract_findings_api_failure_returns_empty(self) -> None:
        """LLM failure should return empty list gracefully."""
        client = MagicMock()
        client.generate.side_effect = RuntimeError("API error")

        extractor = PaperExtractor(client=client)
        results = extractor.extract_key_findings(_SAMPLE_TEXT)

        assert results == []

    # ------------------------------------------------------------------
    # Dual-prompt consistency
    # ------------------------------------------------------------------

    def test_dual_prompt_computes_consistency(self) -> None:
        """Should run both prompts and compute consistency score."""
        client = MagicMock()
        findings = json.dumps({
            "findings": [
                {"claim": "Model achieves 94.2% F1", "supporting_quote": "quote", "confidence": 0.9}
            ]
        })
        client.generate.return_value = _llm_response(findings)

        extractor = PaperExtractor(client=client)
        items, consistency = extractor._dual_prompt_extract(
            _SAMPLE_TEXT,
            "prompt_a: {text}",
            "prompt_b: {text}",
            PaperExtractor._parse_findings,
        )

        # Both prompts return same content → consistency should be 1.0
        assert consistency == 1.0
        assert len(items) >= 1


class TestExtractMethodology:
    """Tests for :meth:`PaperExtractor.extract_methodology`."""

    def test_extract_methodology_returns_object(self) -> None:
        """Should return a MethodologyExtraction on success."""
        client = MagicMock()
        meth_json = json.dumps({
            "approach": "Multi-modal transformer",
            "datasets": ["DocBank", "PubLayNet"],
            "tools": ["PyTorch", "HuggingFace"],
            "eval_metrics": ["F1", "mAP"],
        })
        client.generate.return_value = _llm_response(meth_json)

        extractor = PaperExtractor(client=client)
        result = extractor.extract_methodology(_SAMPLE_TEXT)

        assert isinstance(result, MethodologyExtraction)
        assert result.approach == "Multi-modal transformer"
        assert "DocBank" in result.datasets

    def test_extract_methodology_empty_text_returns_none(self) -> None:
        """Empty text should return None."""
        client = MagicMock()
        extractor = PaperExtractor(client=client)
        assert extractor.extract_methodology("") is None

    def test_extract_methodology_api_failure_returns_none(self) -> None:
        """LLM failure should return None."""
        client = MagicMock()
        client.generate.side_effect = RuntimeError("API error")

        extractor = PaperExtractor(client=client)
        assert extractor.extract_methodology(_SAMPLE_TEXT) is None


class TestExtractResults:
    """Tests for :meth:`PaperExtractor.extract_results`."""

    def test_extract_results_returns_list(self) -> None:
        """Should return a list of ResultExtraction objects."""
        client = MagicMock()
        results_json = json.dumps({
            "results": [
                {"metric_name": "F1", "value": "94.2%", "baseline": "91.1%",
                 "improvement": "+3.1%", "table_reference": "Table 1"},
            ]
        })
        client.generate.return_value = _llm_response(results_json)

        extractor = PaperExtractor(client=client)
        results = extractor.extract_results(_SAMPLE_TEXT)

        assert len(results) == 1
        assert isinstance(results[0], ResultExtraction)
        assert results[0].metric_name == "F1"

    def test_extract_results_empty_text_returns_empty(self) -> None:
        """Empty text should return empty list."""
        client = MagicMock()
        extractor = PaperExtractor(client=client)
        assert extractor.extract_results("") == []

    def test_extract_results_api_failure_returns_empty(self) -> None:
        """LLM failure should return empty list."""
        client = MagicMock()
        client.generate.side_effect = RuntimeError("API error")

        extractor = PaperExtractor(client=client)
        assert extractor.extract_results(_SAMPLE_TEXT) == []


class TestFullPipeline:
    """Tests for :meth:`PaperExtractor.extract` (full pipeline)."""

    def test_extract_combines_all_components(self) -> None:
        """Should run findings, methodology, and results extraction."""
        client = MagicMock()

        # Findings response (called twice for dual-prompt)
        findings_json = json.dumps({
            "findings": [
                {"claim": "Novel approach", "supporting_quote": "We propose a novel", "confidence": 0.9}
            ]
        })
        meth_json = json.dumps({
            "approach": "Transformer", "datasets": [], "tools": [], "eval_metrics": []
        })
        results_json = json.dumps({"results": []})

        # generate() is called 4 times: findings x2, methodology x1, results x1
        client.generate.side_effect = [
            _llm_response(findings_json),
            _llm_response(findings_json),
            _llm_response(meth_json),
            _llm_response(results_json),
        ]

        extractor = PaperExtractor(client=client)
        result = extractor.extract(_SAMPLE_TEXT)

        assert isinstance(result, PaperExtraction)
        assert result.paper_id == "unknown"
        assert len(result.key_findings) >= 1

    def test_extract_uses_metadata_title_as_paper_id(self) -> None:
        """Should use metadata.title as paper_id when available."""
        client = MagicMock()
        findings_json = json.dumps({"findings": []})
        meth_json = json.dumps({
            "approach": "test", "datasets": [], "tools": [], "eval_metrics": []
        })
        results_json = json.dumps({"results": []})

        client.generate.side_effect = [
            _llm_response(findings_json),
            _llm_response(findings_json),
            _llm_response(meth_json),
            _llm_response(results_json),
        ]

        from src.models.schemas import DocumentMetadataSchema
        meta = DocumentMetadataSchema(title="My Paper Title")

        extractor = PaperExtractor(client=client)
        result = extractor.extract(_SAMPLE_TEXT, metadata=meta)

        assert result.paper_id == "My Paper Title"

    def test_extract_low_confidence_flags_review(self) -> None:
        """Papers with low confidence should have needs_review=True."""
        client = MagicMock()
        # Return empty findings → confidence will be low
        client.generate.return_value = _llm_response(json.dumps({"findings": []}))

        extractor = PaperExtractor(client=client)
        result = extractor.extract(_SAMPLE_TEXT)

        # No findings, no methodology, no results → overall 0.0 < 0.7
        assert result.needs_review is True


class TestConfidenceScoring:
    """Tests for confidence scoring helpers."""

    def test_source_grounding_exact_match(self) -> None:
        """Exact substring match should return 1.0."""
        score = PaperExtractor._check_source_grounding(
            "novel approach", "We propose a novel approach to understanding."
        )
        assert score == 1.0

    def test_source_grounding_empty_quote(self) -> None:
        """Empty quote should return 0.0."""
        score = PaperExtractor._check_source_grounding("", "some text")
        assert score == 0.0

    def test_compute_confidence_formula(self) -> None:
        """Should apply 0.4*consistency + 0.4*grounding + 0.2*completeness."""
        result = PaperExtractor._compute_confidence(
            consistency_score=1.0,
            grounding_scores=[1.0, 1.0],
            completeness_score=1.0,
        )
        # 0.4*1.0 + 0.4*1.0 + 0.2*1.0 = 1.0
        assert abs(result - 1.0) < 1e-6

    def test_compute_confidence_partial_scores(self) -> None:
        """Should handle partial scores correctly."""
        result = PaperExtractor._compute_confidence(
            consistency_score=0.5,
            grounding_scores=[0.8, 0.6],
            completeness_score=0.33,
        )
        # 0.4*0.5 + 0.4*0.7 + 0.2*0.33 = 0.2 + 0.28 + 0.066 = 0.546
        expected = 0.4 * 0.5 + 0.4 * 0.7 + 0.2 * 0.33
        assert abs(result - expected) < 1e-6


class TestParsingHelpers:
    """Tests for JSON parsing helpers."""

    def test_parse_findings_dict_format(self) -> None:
        """Should parse {\"findings\": [...]} format."""
        raw = json.dumps({"findings": [{"claim": "test", "supporting_quote": "q"}]})
        result = PaperExtractor._parse_findings(raw)
        assert len(result) == 1
        assert result[0]["claim"] == "test"

    def test_parse_findings_list_format(self) -> None:
        """Should parse bare list format."""
        raw = json.dumps([{"claim": "test", "supporting_quote": "q"}])
        result = PaperExtractor._parse_findings(raw)
        assert len(result) == 1

    def test_parse_findings_invalid_json(self) -> None:
        """Invalid JSON should return empty list."""
        assert PaperExtractor._parse_findings("not json") == []

    def test_strip_markdown_fences(self) -> None:
        """Should extract content from markdown code fences."""
        text = '```json\n{"findings": []}\n```'
        result = PaperExtractor._strip_markdown_fences(text)
        assert result == '{"findings": []}'

    def test_strip_markdown_fences_no_fences(self) -> None:
        """Text without fences should be returned stripped."""
        assert PaperExtractor._strip_markdown_fences('  {"a": 1}  ') == '{"a": 1}'
