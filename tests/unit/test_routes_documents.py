"""Tests for document management endpoints."""

from __future__ import annotations

import io
import uuid
from datetime import datetime, timezone
from unittest.mock import patch

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
    ExtractedTable,
    SectionType,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

API_KEY = "test-api-key-12345"
HEADERS = {"X-API-Key": API_KEY}


def _make_doc(
    doc_id: uuid.UUID | None = None,
    status: DocumentStatus = DocumentStatus.COMPLETED,
    num_pages: int = 2,
    with_sections: bool = True,
    with_tables: bool = True,
) -> DocumentRecord:
    """Create a DocumentRecord for testing."""
    doc_id = doc_id or uuid.uuid4()
    sections = []
    if with_sections:
        sections = [
            DetectedSection(
                section_type=SectionType.ABSTRACT,
                title="Abstract",
                text="Test abstract text.",
                page_start=0,
                page_end=0,
                order_index=0,
            ),
            DetectedSection(
                section_type=SectionType.INTRODUCTION,
                title="Introduction",
                text="Test introduction.",
                page_start=1,
                page_end=1,
                order_index=1,
            ),
        ]
    tables = []
    if with_tables:
        tables = [
            ExtractedTable(
                page_number=1,
                table_index=0,
                headers=["A", "B"],
                rows=[["1", "2"]],
                extraction_method="pdfplumber",
                confidence=0.9,
            )
        ]
    return DocumentRecord(
        id=doc_id,
        filename="test.pdf",
        file_path=f"/tmp/{doc_id}.pdf",
        file_hash="abc123",
        file_size_bytes=1000,
        num_pages=num_pages,
        status=status,
        created_at=datetime.now(timezone.utc),
        metadata=DocumentMetadataSchema(
            title="Test Paper", authors=["Author"], confidence=0.9
        ),
        sections=sections,
        tables=tables,
    )


def _make_store(docs: list[DocumentRecord] | None = None) -> InMemoryDocumentStore:
    """Create a pre-populated InMemoryDocumentStore."""
    store = InMemoryDocumentStore()
    for doc in docs or []:
        store.save(doc)
    return store


# ---------------------------------------------------------------------------
# Upload tests
# ---------------------------------------------------------------------------


class TestUploadDocument:
    """Tests for POST /api/v1/documents/upload."""

    @pytest.mark.asyncio
    async def test_upload_success(self, tmp_path) -> None:
        """Upload returns 201 with doc_id."""
        store = _make_store()
        app.dependency_overrides[verify_api_key] = lambda: API_KEY
        app.dependency_overrides[get_document_store] = lambda: store

        pdf_content = b"%PDF-1.4 minimal test content"
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(pdf_content)

        try:
            with patch("src.api.routes_documents.settings") as mock_settings:
                mock_settings.max_upload_size_mb = 50
                mock_settings.upload_dir = str(tmp_path)

                # Patch Celery task import to force sync fallback
                with patch("src.workers.tasks.process_document", None):
                    with patch("src.workers.tasks._process_document_impl"):
                        transport = ASGITransport(app=app)
                        async with AsyncClient(transport=transport, base_url="http://test") as client:
                            response = await client.post(
                                "/api/v1/documents/upload",
                                files={"file": ("paper.pdf", pdf_content, "application/pdf")},
                                headers=HEADERS,
                            )

            assert response.status_code == 201
            data = response.json()
            assert "id" in data
            assert data["filename"] == "paper.pdf"
            assert data["status"] in ["pending", "completed", "processing"]
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_upload_missing_auth(self) -> None:
        """Upload without API key returns 401."""
        # Don't override verify_api_key — let real auth run
        app.dependency_overrides.pop(verify_api_key, None)
        app.dependency_overrides.pop(get_document_store, None)

        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/documents/upload",
                    files={"file": ("paper.pdf", b"content", "application/pdf")},
                )

            assert response.status_code == 401
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_upload_non_pdf(self) -> None:
        """Non-PDF file returns 400."""
        app.dependency_overrides[verify_api_key] = lambda: API_KEY

        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/documents/upload",
                    files={"file": ("data.csv", b"a,b,c", "text/csv")},
                    headers=HEADERS,
                )

            assert response.status_code == 400
            assert "PDF" in response.json()["detail"]
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_upload_oversized(self, tmp_path) -> None:
        """Oversized file returns 413."""
        app.dependency_overrides[verify_api_key] = lambda: API_KEY

        try:
            with patch("src.api.routes_documents.settings") as mock_settings:
                mock_settings.max_upload_size_mb = 0  # 0 MB = reject everything
                mock_settings.upload_dir = str(tmp_path)

                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    response = await client.post(
                        "/api/v1/documents/upload",
                        files={"file": ("big.pdf", b"%PDF-1.4 " + b"x" * 100, "application/pdf")},
                        headers=HEADERS,
                    )

            assert response.status_code == 413
        finally:
            app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Document detail tests
# ---------------------------------------------------------------------------


class TestGetDocument:
    """Tests for GET /api/v1/documents/{doc_id}."""

    @pytest.mark.asyncio
    async def test_get_document_found(self) -> None:
        """Returns 200 with full document details."""
        doc = _make_doc()
        store = _make_store([doc])
        app.dependency_overrides[verify_api_key] = lambda: API_KEY
        app.dependency_overrides[get_document_store] = lambda: store

        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get(
                    f"/api/v1/documents/{doc.id}", headers=HEADERS
                )

            assert response.status_code == 200
            data = response.json()
            assert data["id"] == str(doc.id)
            assert data["filename"] == "test.pdf"
            assert data["status"] == "completed"
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_get_document_not_found(self) -> None:
        """Returns 404 for unknown doc_id."""
        store = _make_store()
        app.dependency_overrides[verify_api_key] = lambda: API_KEY
        app.dependency_overrides[get_document_store] = lambda: store

        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get(
                    f"/api/v1/documents/{uuid.uuid4()}", headers=HEADERS
                )

            assert response.status_code == 404
        finally:
            app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Document status tests
# ---------------------------------------------------------------------------


class TestGetDocumentStatus:
    """Tests for GET /api/v1/documents/{doc_id}/status."""

    @pytest.mark.asyncio
    async def test_status_returns_fields(self) -> None:
        """Returns status with expected fields."""
        doc = _make_doc(status=DocumentStatus.PROCESSING)
        doc.task_id = "celery-task-123"
        store = _make_store([doc])
        app.dependency_overrides[verify_api_key] = lambda: API_KEY
        app.dependency_overrides[get_document_store] = lambda: store

        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get(
                    f"/api/v1/documents/{doc.id}/status", headers=HEADERS
                )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "processing"
            assert data["document_id"] == str(doc.id)
            assert data["task_id"] == "celery-task-123"
        finally:
            app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# List documents tests
# ---------------------------------------------------------------------------


class TestListDocuments:
    """Tests for GET /api/v1/documents/."""

    @pytest.mark.asyncio
    async def test_list_default_pagination(self) -> None:
        """Returns all documents with default pagination."""
        docs = [_make_doc() for _ in range(3)]
        store = _make_store(docs)
        app.dependency_overrides[verify_api_key] = lambda: API_KEY
        app.dependency_overrides[get_document_store] = lambda: store

        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get(
                    "/api/v1/documents/", headers=HEADERS
                )

            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 3
            assert len(data["documents"]) == 3
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_list_custom_offset_limit(self) -> None:
        """Pagination with custom offset and limit."""
        docs = [_make_doc() for _ in range(5)]
        store = _make_store(docs)
        app.dependency_overrides[verify_api_key] = lambda: API_KEY
        app.dependency_overrides[get_document_store] = lambda: store

        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get(
                    "/api/v1/documents/?offset=2&limit=2", headers=HEADERS
                )

            assert response.status_code == 200
            data = response.json()
            assert len(data["documents"]) == 2
            assert data["offset"] == 2
            assert data["limit"] == 2
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_list_filter_by_status(self) -> None:
        """Filter documents by status."""
        doc1 = _make_doc(status=DocumentStatus.COMPLETED)
        doc2 = _make_doc(status=DocumentStatus.PENDING)
        store = _make_store([doc1, doc2])
        app.dependency_overrides[verify_api_key] = lambda: API_KEY
        app.dependency_overrides[get_document_store] = lambda: store

        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get(
                    "/api/v1/documents/?status=completed", headers=HEADERS
                )

            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 1
            assert data["documents"][0]["status"] == "completed"
        finally:
            app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Delete tests
# ---------------------------------------------------------------------------


class TestDeleteDocument:
    """Tests for DELETE /api/v1/documents/{doc_id}."""

    @pytest.mark.asyncio
    async def test_delete_success(self, tmp_path) -> None:
        """Delete returns 204 and removes the document."""
        doc = _make_doc()
        # Create the actual file so the delete doesn't error
        pdf_file = tmp_path / f"{doc.id}.pdf"
        pdf_file.write_bytes(b"test")
        doc.file_path = str(pdf_file)

        store = _make_store([doc])
        app.dependency_overrides[verify_api_key] = lambda: API_KEY
        app.dependency_overrides[get_document_store] = lambda: store

        try:
            with patch("src.retrieval.vector_store.VectorStore"):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    response = await client.delete(
                        f"/api/v1/documents/{doc.id}", headers=HEADERS
                    )

            assert response.status_code == 204
            assert store.get(doc.id) is None
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_delete_not_found(self) -> None:
        """Delete unknown doc returns 404."""
        store = _make_store()
        app.dependency_overrides[verify_api_key] = lambda: API_KEY
        app.dependency_overrides[get_document_store] = lambda: store

        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.delete(
                    f"/api/v1/documents/{uuid.uuid4()}", headers=HEADERS
                )

            assert response.status_code == 404
        finally:
            app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Sections and tables sub-resources
# ---------------------------------------------------------------------------


class TestGetSections:
    """Tests for GET /api/v1/documents/{doc_id}/sections."""

    @pytest.mark.asyncio
    async def test_returns_sections(self) -> None:
        """Returns sections list for a document."""
        doc = _make_doc(with_sections=True)
        store = _make_store([doc])
        app.dependency_overrides[verify_api_key] = lambda: API_KEY
        app.dependency_overrides[get_document_store] = lambda: store

        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get(
                    f"/api/v1/documents/{doc.id}/sections", headers=HEADERS
                )

            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 2
            assert len(data["sections"]) == 2
        finally:
            app.dependency_overrides.clear()


class TestGetTables:
    """Tests for GET /api/v1/documents/{doc_id}/tables."""

    @pytest.mark.asyncio
    async def test_returns_tables(self) -> None:
        """Returns tables list for a document."""
        doc = _make_doc(with_tables=True)
        store = _make_store([doc])
        app.dependency_overrides[verify_api_key] = lambda: API_KEY
        app.dependency_overrides[get_document_store] = lambda: store

        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get(
                    f"/api/v1/documents/{doc.id}/tables", headers=HEADERS
                )

            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 1
            assert len(data["tables"]) == 1
        finally:
            app.dependency_overrides.clear()
