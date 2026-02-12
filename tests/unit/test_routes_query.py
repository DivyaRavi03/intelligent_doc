"""Tests for query, search, compare, and summary endpoints."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.auth import verify_api_key
from src.api.stores import (
    DocumentRecord,
    InMemoryDocumentStore,
    get_document_store,
)
from src.main import app
from src.models.schemas import (
    DetectedSection,
    DocumentMetadataSchema,
    DocumentStatus,
    SectionType,
)

API_KEY = "test-api-key-12345"
HEADERS = {"X-API-Key": API_KEY}


def _make_completed_doc(doc_id: uuid.UUID | None = None) -> DocumentRecord:
    """Create a completed document record."""
    doc_id = doc_id or uuid.uuid4()
    return DocumentRecord(
        id=doc_id,
        filename="test.pdf",
        file_path=f"/tmp/{doc_id}.pdf",
        file_hash="abc",
        file_size_bytes=1000,
        num_pages=2,
        status=DocumentStatus.COMPLETED,
        created_at=datetime.now(timezone.utc),
        metadata=DocumentMetadataSchema(
            title="Test Paper", authors=["Author"], confidence=0.9
        ),
        sections=[
            DetectedSection(
                section_type=SectionType.ABSTRACT,
                title="Abstract",
                text="This paper discusses deep learning approaches.",
                page_start=0,
                page_end=0,
                order_index=0,
            ),
        ],
    )


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------


class TestQueryEndpoint:
    """Tests for POST /api/v1/query."""

    @pytest.mark.asyncio
    async def test_query_returns_response(self) -> None:
        """Successful query returns QAResponse."""
        from src.models.schemas import QAResponse

        mock_qa = MagicMock()
        mock_qa.answer.return_value = QAResponse(
            query="What is deep learning?",
            answer="Test answer",
            citations=[],
            claim_verifications=[],
            faithfulness_score=0.95,
            flagged_claims=[],
        )

        app.dependency_overrides[verify_api_key] = lambda: API_KEY
        try:
            with patch("src.api.routes_query._get_qa_engine", return_value=mock_qa):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    response = await client.post(
                        "/api/v1/query",
                        json={"query": "What is deep learning?"},
                        headers=HEADERS,
                    )

            assert response.status_code == 200
            data = response.json()
            assert "answer" in data
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_query_missing_auth(self) -> None:
        """Query without auth returns 401."""
        app.dependency_overrides.pop(verify_api_key, None)
        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/query",
                    json={"query": "What is deep learning?"},
                )

            assert response.status_code == 401
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_query_empty_query(self) -> None:
        """Empty query string returns 422 validation error."""
        app.dependency_overrides[verify_api_key] = lambda: API_KEY
        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/query",
                    json={"query": ""},
                    headers=HEADERS,
                )

            assert response.status_code == 422
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_query_engine_error(self) -> None:
        """QA engine exception returns 500."""
        mock_qa = MagicMock()
        mock_qa.answer.side_effect = RuntimeError("Engine broken")

        app.dependency_overrides[verify_api_key] = lambda: API_KEY
        try:
            with patch("src.api.routes_query._get_qa_engine", return_value=mock_qa):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    response = await client.post(
                        "/api/v1/query",
                        json={"query": "test"},
                        headers=HEADERS,
                    )

            assert response.status_code == 500
        finally:
            app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# POST /search
# ---------------------------------------------------------------------------


class TestSearchEndpoint:
    """Tests for POST /api/v1/search."""

    @pytest.mark.asyncio
    async def test_search_returns_results(self) -> None:
        """Successful search returns SearchResponse."""
        mock_result = MagicMock()
        mock_result.chunk_id = "chunk-1"
        mock_result.text = "Found text"
        mock_result.final_score = 0.85
        mock_result.rrf_score = 0.80
        mock_result.paper_id = "paper-1"
        mock_result.section_type = "abstract"
        mock_result.page_numbers = [1]

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [mock_result]

        app.dependency_overrides[verify_api_key] = lambda: API_KEY
        try:
            with patch("src.api.routes_query._get_retriever", return_value=mock_retriever):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    response = await client.post(
                        "/api/v1/search",
                        json={"query": "deep learning"},
                        headers=HEADERS,
                    )

            assert response.status_code == 200
            data = response.json()
            assert data["query"] == "deep learning"
            assert len(data["results"]) == 1
            assert data["results"][0]["score"] == 0.85
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_search_with_filters(self) -> None:
        """Search with paper_id and section_type filters."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []

        app.dependency_overrides[verify_api_key] = lambda: API_KEY
        try:
            with patch("src.api.routes_query._get_retriever", return_value=mock_retriever):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    response = await client.post(
                        "/api/v1/search",
                        json={
                            "query": "transformers",
                            "paper_id": "paper-1",
                            "section_type": "methodology",
                            "top_k": 5,
                            "alpha": 0.7,
                        },
                        headers=HEADERS,
                    )

            assert response.status_code == 200
            mock_retriever.retrieve.assert_called_once_with(
                query="transformers",
                top_k=5,
                alpha=0.7,
                paper_id="paper-1",
                section_type="methodology",
            )
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_search_error(self) -> None:
        """Retriever exception returns 500."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.side_effect = RuntimeError("Search broken")

        app.dependency_overrides[verify_api_key] = lambda: API_KEY
        try:
            with patch("src.api.routes_query._get_retriever", return_value=mock_retriever):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    response = await client.post(
                        "/api/v1/search",
                        json={"query": "test"},
                        headers=HEADERS,
                    )

            assert response.status_code == 500
        finally:
            app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# POST /compare
# ---------------------------------------------------------------------------


class TestCompareEndpoint:
    """Tests for POST /api/v1/compare."""

    @pytest.mark.asyncio
    async def test_compare_returns_comparison(self) -> None:
        """Successful compare returns CompareResponse."""
        doc1 = _make_completed_doc()
        doc2 = _make_completed_doc()
        store = InMemoryDocumentStore()
        store.save(doc1)
        store.save(doc2)

        mock_client = MagicMock()
        mock_client.generate.return_value = MagicMock(
            content="Paper 1 focuses on X.\n- Difference A\n- Difference B"
        )

        app.dependency_overrides[verify_api_key] = lambda: API_KEY
        app.dependency_overrides[get_document_store] = lambda: store

        try:
            with patch("src.api.routes_query._get_client", return_value=mock_client):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    response = await client.post(
                        "/api/v1/compare",
                        json={
                            "paper_ids": [str(doc1.id), str(doc2.id)],
                            "aspect": "methodology",
                        },
                        headers=HEADERS,
                    )

            assert response.status_code == 200
            data = response.json()
            assert data["aspect"] == "methodology"
            assert str(doc1.id) in data["papers"]
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_compare_too_few_papers(self) -> None:
        """Compare with fewer than 2 papers returns 422."""
        app.dependency_overrides[verify_api_key] = lambda: API_KEY
        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/compare",
                    json={"paper_ids": ["one"], "aspect": "methodology"},
                    headers=HEADERS,
                )

            assert response.status_code == 422
        finally:
            app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# GET /documents/{doc_id}/summary/{level}
# ---------------------------------------------------------------------------


class TestSummaryEndpoint:
    """Tests for GET /api/v1/documents/{doc_id}/summary/{level}."""

    @pytest.mark.asyncio
    async def test_summary_returns_result(self) -> None:
        """Successful summary at 'abstract' level."""
        doc = _make_completed_doc()
        store = InMemoryDocumentStore()
        store.save(doc)

        mock_summarizer = MagicMock()
        mock_summarizer.summarize.return_value = MagicMock(
            paper_id=str(doc.id),
            level="abstract",
            summary="This paper is about...",
            word_count=10,
            sections_used=["abstract"],
            model_dump=lambda **kw: {
                "paper_id": str(doc.id),
                "level": "abstract",
                "summary": "This paper is about...",
                "word_count": 10,
                "sections_used": ["abstract"],
            },
        )

        app.dependency_overrides[verify_api_key] = lambda: API_KEY
        app.dependency_overrides[get_document_store] = lambda: store

        try:
            with patch("src.api.routes_query._get_summarizer", return_value=mock_summarizer):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    response = await client.get(
                        f"/api/v1/documents/{doc.id}/summary/abstract",
                        headers=HEADERS,
                    )

            assert response.status_code == 200
            data = response.json()
            assert "summary" in data
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_summary_doc_not_found(self) -> None:
        """Summary for unknown doc returns 404."""
        store = InMemoryDocumentStore()
        app.dependency_overrides[verify_api_key] = lambda: API_KEY
        app.dependency_overrides[get_document_store] = lambda: store

        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get(
                    f"/api/v1/documents/{uuid.uuid4()}/summary/abstract",
                    headers=HEADERS,
                )

            assert response.status_code == 404
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_summary_invalid_level(self) -> None:
        """Invalid summary level returns 422."""
        doc = _make_completed_doc()
        store = InMemoryDocumentStore()
        store.save(doc)
        app.dependency_overrides[verify_api_key] = lambda: API_KEY
        app.dependency_overrides[get_document_store] = lambda: store

        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get(
                    f"/api/v1/documents/{doc.id}/summary/mega_detailed",
                    headers=HEADERS,
                )

            assert response.status_code == 422
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_summary_not_completed(self) -> None:
        """Summary for non-completed doc returns 400."""
        doc = _make_completed_doc()
        doc.status = DocumentStatus.PROCESSING
        store = InMemoryDocumentStore()
        store.save(doc)
        app.dependency_overrides[verify_api_key] = lambda: API_KEY
        app.dependency_overrides[get_document_store] = lambda: store

        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get(
                    f"/api/v1/documents/{doc.id}/summary/abstract",
                    headers=HEADERS,
                )

            assert response.status_code == 400
        finally:
            app.dependency_overrides.clear()
