"""Integration tests for the full document processing pipeline.

Tests end-to-end flows from API endpoint to response, with LLM calls
mocked at the GeminiClient boundary.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.stores import InMemoryDocumentStore
from src.main import app
from src.models.schemas import (
    DetectedSection,
    DocumentStatus,
    LLMResponse,
    SectionType,
)

API_KEY = "test-api-key"


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


def _make_completed_doc(doc_id: uuid.UUID | None = None) -> MagicMock:
    """Create a mock completed DocumentRecord."""
    doc = MagicMock()
    doc.id = doc_id or uuid.uuid4()
    doc.status = DocumentStatus.COMPLETED
    doc.sections = [
        DetectedSection(
            section_type=SectionType.ABSTRACT,
            title="Abstract",
            text="This paper presents a novel deep learning approach to NLP tasks.",
            page_start=0,
            order_index=0,
        ),
        DetectedSection(
            section_type=SectionType.METHODOLOGY,
            title="Methodology",
            text="We use a transformer encoder with multi-head attention.",
            page_start=1,
            order_index=1,
        ),
        DetectedSection(
            section_type=SectionType.RESULTS,
            title="Results",
            text="Our model achieves 95.3% F1 score on the benchmark dataset.",
            page_start=2,
            order_index=2,
        ),
    ]
    return doc


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    @pytest.mark.asyncio
    async def test_health_endpoint_returns_200(self) -> None:
        """GET /api/v1/admin/health should return 200."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/admin/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("ok", "healthy")


class TestCompareIntegration:
    """Integration tests for the compare endpoint with CrossPaperAnalyzer."""

    @pytest.mark.asyncio
    async def test_compare_returns_200_with_valid_papers(self) -> None:
        """POST /api/v1/compare should return a comparison."""
        doc_id1 = uuid.uuid4()
        doc_id2 = uuid.uuid4()
        store = InMemoryDocumentStore()
        store.save(_make_completed_doc(doc_id1))
        store.save(_make_completed_doc(doc_id2))

        mock_client = MagicMock()
        mock_client.generate.return_value = _llm_response(
            '{"comparison_table": [], "agreements": ["Both use DL"], '
            '"contradictions": ["Different models"], '
            '"synthesis": "Both papers apply deep learning."}'
        )

        from src.api.auth import verify_api_key
        from src.api.stores import get_document_store

        app.dependency_overrides[verify_api_key] = lambda: API_KEY
        app.dependency_overrides[get_document_store] = lambda: store

        try:
            with patch("src.api.routes_query._get_cross_paper_analyzer") as mock_get:
                from src.llm.cross_paper import CrossPaperAnalyzer
                analyzer = CrossPaperAnalyzer(client=mock_client, store=store)
                mock_get.return_value = analyzer

                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    response = await client.post(
                        "/api/v1/compare",
                        json={
                            "paper_ids": [str(doc_id1), str(doc_id2)],
                            "aspect": "methodology",
                        },
                        headers={"X-API-Key": API_KEY},
                    )

            assert response.status_code == 200
            data = response.json()
            assert data["aspect"] == "methodology"
            assert "comparison_text" in data
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_compare_with_missing_papers(self) -> None:
        """Compare with non-existent papers should still return 200."""
        store = InMemoryDocumentStore()
        mock_client = MagicMock()
        mock_client.generate.return_value = _llm_response(
            '{"comparison_table": [], "agreements": [], '
            '"contradictions": [], "synthesis": "Limited comparison."}'
        )

        from src.api.auth import verify_api_key
        from src.api.stores import get_document_store

        app.dependency_overrides[verify_api_key] = lambda: API_KEY
        app.dependency_overrides[get_document_store] = lambda: store

        try:
            with patch("src.api.routes_query._get_cross_paper_analyzer") as mock_get:
                from src.llm.cross_paper import CrossPaperAnalyzer
                analyzer = CrossPaperAnalyzer(client=mock_client, store=store)
                mock_get.return_value = analyzer

                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    response = await client.post(
                        "/api/v1/compare",
                        json={
                            "paper_ids": [str(uuid.uuid4()), str(uuid.uuid4())],
                            "aspect": "results",
                        },
                        headers={"X-API-Key": API_KEY},
                    )

            assert response.status_code == 200
        finally:
            app.dependency_overrides.clear()


class TestSummaryIntegration:
    """Integration tests for the summary endpoint."""

    @pytest.mark.asyncio
    async def test_summary_returns_result(self) -> None:
        """GET /documents/{id}/summary/one_line should return a summary."""
        from src.api.auth import verify_api_key
        from src.api.stores import get_document_store
        from src.models.schemas import SummaryLevel, SummaryResult

        doc_id = uuid.uuid4()
        store = InMemoryDocumentStore()
        store.save(_make_completed_doc(doc_id))

        mock_summarizer = MagicMock()
        mock_summarizer.summarize.return_value = SummaryResult(
            paper_id=str(doc_id),
            level=SummaryLevel.ONE_LINE,
            summary="A deep learning approach to NLP.",
            word_count=7,
            sections_used=["abstract"],
        )

        app.dependency_overrides[verify_api_key] = lambda: API_KEY
        app.dependency_overrides[get_document_store] = lambda: store

        try:
            with patch("src.api.routes_query._get_summarizer", return_value=mock_summarizer):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    response = await client.get(
                        f"/api/v1/documents/{doc_id}/summary/one_line",
                        headers={"X-API-Key": API_KEY},
                    )

            assert response.status_code == 200
            data = response.json()
            assert "summary" in data
            assert data["level"] == "one_line"
        finally:
            app.dependency_overrides.clear()


class TestQueryIntegration:
    """Integration tests for the query endpoint."""

    @pytest.mark.asyncio
    async def test_query_returns_answer(self) -> None:
        """POST /api/v1/query should return a QA response."""
        from src.api.auth import verify_api_key
        from src.models.schemas import QAResponse

        mock_qa = MagicMock()
        mock_qa.answer.return_value = QAResponse(
            query="What is the F1 score?",
            answer="The model achieves 95.3% F1 [1].",
            faithfulness_score=0.9,
        )

        app.dependency_overrides[verify_api_key] = lambda: API_KEY

        try:
            with patch("src.api.routes_query._get_qa_engine", return_value=mock_qa):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    response = await client.post(
                        "/api/v1/query",
                        json={"query": "What is the F1 score?"},
                        headers={"X-API-Key": API_KEY},
                    )

            assert response.status_code == 200
            data = response.json()
            assert "answer" in data
            assert data["query"] == "What is the F1 score?"
        finally:
            app.dependency_overrides.clear()
