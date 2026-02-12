"""Unit tests for the citation-tracked question-answering engine."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.llm.qa_engine import QAEngine
from src.models.schemas import (
    Citation,
    ClaimVerification,
    LLMResponse,
    QAResponse,
)
from src.retrieval.hybrid_retriever import RankedResult


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


def _ranked(
    chunk_id: str,
    text: str = "sample passage",
    paper_id: str = "paper-001",
    section_type: str = "introduction",
    score: float = 0.5,
) -> RankedResult:
    """Helper to create a minimal RankedResult."""
    return RankedResult(
        chunk_id=chunk_id,
        text=text,
        paper_id=paper_id,
        section_type=section_type,
        rrf_score=score,
        final_score=score,
    )


class TestContextBuilding:
    """Tests for :meth:`QAEngine._build_context`."""

    def test_build_context_formats_sources(self) -> None:
        """Should format chunks with [1] [2] markers."""
        engine = QAEngine()
        chunks = [
            _ranked("c1", text="First passage"),
            _ranked("c2", text="Second passage"),
        ]

        context, citations = engine._build_context(chunks)

        assert "Source [1]:" in context
        assert "Source [2]:" in context
        assert "First passage" in context
        assert len(citations) == 2

    def test_build_context_creates_citations(self) -> None:
        """Should create Citation objects with correct indices."""
        engine = QAEngine()
        chunks = [_ranked("c1", text="passage text", paper_id="p1")]

        _, citations = engine._build_context(chunks)

        assert len(citations) == 1
        assert isinstance(citations[0], Citation)
        assert citations[0].source_index == 1
        assert citations[0].chunk_id == "c1"
        assert citations[0].paper_id == "p1"

    def test_build_context_truncates_snippets(self) -> None:
        """Citation text_snippet should be truncated to 200 chars."""
        engine = QAEngine()
        long_text = "x" * 500
        chunks = [_ranked("c1", text=long_text)]

        _, citations = engine._build_context(chunks)

        assert len(citations[0].text_snippet) == 200


class TestAnswerGeneration:
    """Tests for :meth:`QAEngine.answer` and :meth:`QAEngine._generate`."""

    def test_answer_empty_query_returns_empty(self) -> None:
        """Empty query should return empty answer."""
        engine = QAEngine()
        result = engine.answer("")

        assert isinstance(result, QAResponse)
        assert result.answer == ""

    def test_answer_no_retriever_returns_no_context(self) -> None:
        """No retriever should return 'no relevant context' message."""
        engine = QAEngine(retriever=None)
        result = engine.answer("What is the main finding?")

        assert "No relevant context" in result.answer

    def test_answer_with_retriever_returns_response(self) -> None:
        """Should retrieve, generate, and verify on valid query."""
        client = MagicMock()
        retriever = MagicMock()

        retriever.retrieve.return_value = [
            _ranked("c1", text="The main finding is X."),
        ]

        # First call: generate answer; second call: verify claims
        answer_json = json.dumps({"answer": "The main finding is X. [1]"})
        verify_json = json.dumps({
            "verifications": [
                {"claim": "The main finding is X.", "cited_source_index": 1,
                 "status": "SUPPORTED", "explanation": "Matches source"}
            ]
        })
        client.generate.side_effect = [
            _llm_response(answer_json),
            _llm_response(verify_json),
        ]

        engine = QAEngine(client=client, retriever=retriever)
        result = engine.answer("What is the main finding?")

        assert isinstance(result, QAResponse)
        assert "main finding" in result.answer
        assert result.faithfulness_score > 0.0

    def test_generate_extracts_answer_from_json(self) -> None:
        """Should extract 'answer' field from JSON response."""
        client = MagicMock()
        client.generate.return_value = _llm_response(
            json.dumps({"answer": "Extracted answer text"})
        )

        engine = QAEngine(client=client)
        result = engine._generate("query", "context")

        assert result == "Extracted answer text"

    def test_generate_returns_raw_on_non_json(self) -> None:
        """Should return raw content if not valid JSON."""
        client = MagicMock()
        client.generate.return_value = _llm_response("Plain text answer")

        engine = QAEngine(client=client)
        result = engine._generate("query", "context")

        assert result == "Plain text answer"

    def test_generate_api_failure_returns_fallback(self) -> None:
        """LLM failure should return fallback message."""
        client = MagicMock()
        client.generate.side_effect = RuntimeError("API error")

        engine = QAEngine(client=client)
        result = engine._generate("query", "context")

        assert "Unable to generate" in result


class TestVerification:
    """Tests for :meth:`QAEngine._verify` and verification parsing."""

    def test_verify_all_supported(self) -> None:
        """All supported claims should yield faithfulness = 1.0."""
        client = MagicMock()
        verify_json = json.dumps({
            "verifications": [
                {"claim": "Claim A", "cited_source_index": 1,
                 "status": "SUPPORTED", "explanation": "Match"},
                {"claim": "Claim B", "cited_source_index": 2,
                 "status": "SUPPORTED", "explanation": "Match"},
            ]
        })
        client.generate.return_value = _llm_response(verify_json)

        citations = [
            Citation(source_index=1, chunk_id="c1", text_snippet="text", paper_id="p1", section_type="introduction"),
            Citation(source_index=2, chunk_id="c2", text_snippet="text", paper_id="p1", section_type="methodology"),
        ]

        engine = QAEngine(client=client)
        verifications, score, flagged = engine._verify(
            "Claim A [1]. Claim B [2].", "context", citations
        )

        assert score == 1.0
        assert flagged == []
        assert len(verifications) == 2

    def test_verify_mixed_support(self) -> None:
        """Mixed support should yield partial faithfulness."""
        client = MagicMock()
        verify_json = json.dumps({
            "verifications": [
                {"claim": "Claim A", "cited_source_index": 1,
                 "status": "SUPPORTED", "explanation": "OK"},
                {"claim": "Claim B", "cited_source_index": 2,
                 "status": "NOT_SUPPORTED", "explanation": "Not found"},
            ]
        })
        client.generate.return_value = _llm_response(verify_json)

        citations = [
            Citation(source_index=1, chunk_id="c1", text_snippet="text", paper_id="p1", section_type="introduction"),
            Citation(source_index=2, chunk_id="c2", text_snippet="text", paper_id="p1", section_type="methodology"),
        ]

        engine = QAEngine(client=client)
        verifications, score, flagged = engine._verify(
            "Claim A [1]. Claim B [2].", "context", citations
        )

        assert score == 0.5
        assert flagged == ["Claim B"]

    def test_verify_empty_answer(self) -> None:
        """Empty answer should return empty verifications."""
        engine = QAEngine()
        verifications, score, flagged = engine._verify("", "ctx", [])
        assert verifications == []
        assert score == 0.0

    def test_verify_api_failure(self) -> None:
        """LLM failure during verification should return empty."""
        client = MagicMock()
        client.generate.side_effect = RuntimeError("API error")

        engine = QAEngine(client=client)
        citations = [Citation(source_index=1, chunk_id="c1", text_snippet="t", paper_id="p1", section_type="introduction")]
        verifications, score, flagged = engine._verify(
            "Some claim [1].", "context", citations
        )

        assert verifications == []
        assert score == 0.0

    def test_parse_verifications_valid_json(self) -> None:
        """Should parse standard verification JSON."""
        raw = json.dumps({
            "verifications": [
                {"claim": "test", "cited_source_index": 1,
                 "status": "SUPPORTED", "explanation": "matches"}
            ]
        })
        result = QAEngine._parse_verifications(raw)
        assert len(result) == 1
        assert isinstance(result[0], ClaimVerification)
        assert result[0].status == "SUPPORTED"

    def test_parse_verifications_invalid_json(self) -> None:
        """Invalid JSON should return empty list."""
        assert QAEngine._parse_verifications("bad json") == []


class TestClaimSplitting:
    """Tests for :meth:`QAEngine._split_claims`."""

    def test_split_claims_extracts_citations(self) -> None:
        """Should extract sentence and cited source index."""
        engine = QAEngine()
        claims = engine._split_claims(
            "The model achieves 94% F1 [1]. It uses transformers [2]."
        )

        assert len(claims) == 2
        assert claims[0] == ("The model achieves 94% F1 [1].", 1)
        assert claims[1] == ("It uses transformers [2].", 2)

    def test_split_claims_uncited_sentence(self) -> None:
        """Uncited sentences should have None as source index."""
        engine = QAEngine()
        claims = engine._split_claims("This has no citation.")

        assert len(claims) == 1
        assert claims[0] == ("This has no citation.", None)

    def test_split_claims_empty_string(self) -> None:
        """Empty string should return empty list."""
        engine = QAEngine()
        assert engine._split_claims("") == []

    def test_split_claims_multiple_citations_takes_first(self) -> None:
        """Sentence with multiple citations should use the first."""
        engine = QAEngine()
        claims = engine._split_claims("Both sources agree [1][2].")

        assert len(claims) == 1
        assert claims[0][1] == 1


class TestRetrieval:
    """Tests for :meth:`QAEngine._retrieve_chunks`."""

    def test_retrieve_chunks_no_retriever(self) -> None:
        """Should return empty list when no retriever is set."""
        engine = QAEngine(retriever=None)
        assert engine._retrieve_chunks("query", None, 5) == []

    def test_retrieve_chunks_no_paper_ids(self) -> None:
        """Should call retriever.retrieve directly without paper filter."""
        retriever = MagicMock()
        retriever.retrieve.return_value = [_ranked("c1")]

        engine = QAEngine(retriever=retriever)
        results = engine._retrieve_chunks("query", None, 5)

        assert len(results) == 1
        retriever.retrieve.assert_called_once_with("query", top_k=5)

    def test_retrieve_chunks_deduplicates(self) -> None:
        """Should deduplicate chunks across paper IDs."""
        retriever = MagicMock()
        chunk = _ranked("c1", score=0.8)
        retriever.retrieve.return_value = [chunk]

        engine = QAEngine(retriever=retriever)
        results = engine._retrieve_chunks("query", ["p1", "p2"], top_k=5)

        # Same chunk_id from both papers → deduplicated to 1
        assert len(results) == 1
