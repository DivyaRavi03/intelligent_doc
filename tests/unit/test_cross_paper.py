"""Unit tests for the cross-paper comparison analyzer."""

from __future__ import annotations

import json
import uuid
from unittest.mock import MagicMock

import pytest

from src.api.stores import InMemoryDocumentStore
from src.llm.cross_paper import CrossPaperAnalyzer
from src.models.schemas import (
    ComparisonTableRow,
    CrossPaperComparison,
    DetectedSection,
    DocumentStatus,
    LLMResponse,
    SectionType,
)


def _llm_response(content: str) -> LLMResponse:
    """Helper to create a minimal LLMResponse."""
    return LLMResponse(
        content=content,
        input_tokens=10,
        output_tokens=5,
        latency_ms=100.0,
        model="gemini-1.5-pro",
        cached=False,
        cost_usd=0.0001,
    )


def _make_completed_doc(doc_id: uuid.UUID) -> MagicMock:
    """Create a mock DocumentRecord with sections."""
    doc = MagicMock()
    doc.id = doc_id
    doc.status = DocumentStatus.COMPLETED
    doc.sections = [
        DetectedSection(
            section_type=SectionType.ABSTRACT,
            title="Abstract",
            text="This paper presents a novel method for document understanding.",
            page_start=0,
            order_index=0,
        ),
        DetectedSection(
            section_type=SectionType.METHODOLOGY,
            title="Methodology",
            text="We use transformer-based models with attention mechanisms.",
            page_start=1,
            order_index=1,
        ),
    ]
    return doc


_COMPARISON_JSON = json.dumps({
    "comparison_table": [
        {"aspect": "approach", "papers": {"p1": "transformers", "p2": "CNNs"}},
    ],
    "agreements": ["Both use deep learning"],
    "contradictions": ["Different architectures"],
    "synthesis": "Paper 1 uses transformers while Paper 2 uses CNNs.",
})


class TestCrossPaperAnalyzer:
    """Tests for :class:`CrossPaperAnalyzer`."""

    def test_compare_returns_cross_paper_comparison(self) -> None:
        """compare_papers should return a CrossPaperComparison instance."""
        doc_id1 = uuid.uuid4()
        doc_id2 = uuid.uuid4()
        store = InMemoryDocumentStore()
        store.save(_make_completed_doc(doc_id1))
        store.save(_make_completed_doc(doc_id2))

        client = MagicMock()
        client.generate.return_value = _llm_response(_COMPARISON_JSON)

        analyzer = CrossPaperAnalyzer(client=client, store=store)
        result = analyzer.compare_papers([str(doc_id1), str(doc_id2)], "methodology")

        assert isinstance(result, CrossPaperComparison)
        assert result.aspect == "methodology"
        assert len(result.paper_ids) == 2

    def test_gather_context_from_store(self) -> None:
        """_gather_context should retrieve text from stored documents."""
        doc_id = uuid.uuid4()
        store = InMemoryDocumentStore()
        store.save(_make_completed_doc(doc_id))

        client = MagicMock()
        analyzer = CrossPaperAnalyzer(client=client, store=store)
        context = analyzer._gather_context([str(doc_id)])

        assert str(doc_id) in context
        assert "novel method" in context[str(doc_id)]

    def test_gather_context_unavailable_paper(self) -> None:
        """Missing papers should get 'Content unavailable'."""
        store = InMemoryDocumentStore()
        client = MagicMock()
        analyzer = CrossPaperAnalyzer(client=client, store=store)
        fake_id = str(uuid.uuid4())
        context = analyzer._gather_context([fake_id])

        assert context[fake_id] == "Content unavailable"

    def test_parse_comparison_valid_json(self) -> None:
        """Valid JSON should be parsed into structured fields."""
        client = MagicMock()
        store = InMemoryDocumentStore()
        analyzer = CrossPaperAnalyzer(client=client, store=store)

        result = analyzer._parse_comparison(
            _COMPARISON_JSON, ["p1", "p2"], "methodology", "gemini-1.5-pro"
        )

        assert len(result.comparison_table) == 1
        assert result.comparison_table[0].aspect == "approach"
        assert result.agreements == ["Both use deep learning"]
        assert result.contradictions == ["Different architectures"]
        assert "transformers" in result.synthesis

    def test_parse_comparison_malformed_json_fallback(self) -> None:
        """Malformed JSON should fall back to raw text as synthesis."""
        client = MagicMock()
        store = InMemoryDocumentStore()
        analyzer = CrossPaperAnalyzer(client=client, store=store)

        result = analyzer._parse_comparison(
            "This is not JSON at all", ["p1", "p2"], "results", "model"
        )

        assert result.synthesis == "This is not JSON at all"
        assert result.comparison_table == []
        assert result.agreements == []

    def test_compare_calls_generate(self) -> None:
        """compare_papers should call client.generate exactly once."""
        doc_id1 = uuid.uuid4()
        doc_id2 = uuid.uuid4()
        store = InMemoryDocumentStore()
        store.save(_make_completed_doc(doc_id1))
        store.save(_make_completed_doc(doc_id2))

        client = MagicMock()
        client.generate.return_value = _llm_response(_COMPARISON_JSON)

        analyzer = CrossPaperAnalyzer(client=client, store=store)
        analyzer.compare_papers([str(doc_id1), str(doc_id2)])

        client.generate.assert_called_once()

    def test_compare_with_custom_aspect(self) -> None:
        """Custom aspects should be passed through to the result."""
        doc_id1 = uuid.uuid4()
        doc_id2 = uuid.uuid4()
        store = InMemoryDocumentStore()
        store.save(_make_completed_doc(doc_id1))
        store.save(_make_completed_doc(doc_id2))

        client = MagicMock()
        client.generate.return_value = _llm_response(_COMPARISON_JSON)

        analyzer = CrossPaperAnalyzer(client=client, store=store)
        result = analyzer.compare_papers([str(doc_id1), str(doc_id2)], "results")

        assert result.aspect == "results"

    def test_empty_paper_ids_raises(self) -> None:
        """Fewer than 2 paper IDs should raise ValueError."""
        client = MagicMock()
        store = InMemoryDocumentStore()
        analyzer = CrossPaperAnalyzer(client=client, store=store)

        with pytest.raises(ValueError, match="At least 2"):
            analyzer.compare_papers(["only-one"])

    def test_parse_comparison_with_markdown_fences(self) -> None:
        """JSON wrapped in markdown fences should be parsed correctly."""
        client = MagicMock()
        store = InMemoryDocumentStore()
        analyzer = CrossPaperAnalyzer(client=client, store=store)

        fenced = f"```json\n{_COMPARISON_JSON}\n```"
        result = analyzer._parse_comparison(fenced, ["p1", "p2"], "methodology", "model")

        assert len(result.comparison_table) == 1
        assert result.agreements == ["Both use deep learning"]
