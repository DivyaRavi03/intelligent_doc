"""Document management endpoints.

Provides upload, retrieval, listing, deletion, and sub-resource access
for processed research papers.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile

from src.api.auth import verify_api_key
from src.api.rate_limiter import UPLOAD_LIMIT, limiter
from src.api.stores import (
    DocumentRecord,
    InMemoryDocumentStore,
    get_document_store,
    update_metrics,
)
from src.config import settings
from src.models.schemas import (
    DocumentDetailResponse,
    DocumentListItem,
    DocumentListResponse,
    DocumentStatus,
    DocumentUploadResponse,
    TaskStatusResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


# ------------------------------------------------------------------
# POST /upload
# ------------------------------------------------------------------


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=201,
    summary="Upload a PDF document",
    description="Upload a PDF file for processing. Returns a document ID and task ID.",
)
@limiter.limit(UPLOAD_LIMIT)
async def upload_document(
    request: Request,
    file: UploadFile,
    store: InMemoryDocumentStore = Depends(get_document_store),
    api_key: str = Depends(verify_api_key),
) -> DocumentUploadResponse:
    """Upload a PDF and queue it for asynchronous processing."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    content = await file.read()
    file_size = len(content)
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if file_size > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {settings.max_upload_size_mb}MB limit",
        )

    file_hash = hashlib.sha256(content).hexdigest()
    doc_id = uuid.uuid4()

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    save_path = upload_dir / f"{doc_id}.pdf"
    save_path.write_bytes(content)

    doc = DocumentRecord(
        id=doc_id,
        filename=file.filename,
        file_path=str(save_path),
        file_hash=file_hash,
        file_size_bytes=file_size,
    )

    # Try Celery, fall back to sync
    task_id: str | None = None
    try:
        from src.workers.tasks import process_document

        if process_document is not None:
            result = process_document.delay(str(doc_id))
            task_id = result.id
        else:
            raise RuntimeError("Celery task not available")
    except Exception:
        logger.info("Celery unavailable, processing synchronously: %s", doc_id)
        from src.workers.tasks import _process_document_impl

        doc.status = DocumentStatus.PROCESSING
        store.save(doc)
        try:
            _process_document_impl(str(doc_id), save_path)
        except Exception as exc:
            logger.warning("Sync processing failed for %s: %s", doc_id, exc)

    doc.task_id = task_id
    store.save(doc)
    update_metrics("total_documents", 1)

    # Re-read to get latest status after potential sync processing
    doc = store.get(doc_id) or doc

    return DocumentUploadResponse(
        id=doc.id,
        filename=doc.filename,
        status=doc.status,
        num_pages=doc.num_pages,
        task_id=task_id,
        created_at=doc.created_at,
    )


# ------------------------------------------------------------------
# GET /{doc_id}
# ------------------------------------------------------------------


@router.get(
    "/{doc_id}",
    response_model=DocumentDetailResponse,
    summary="Get document details",
    description="Retrieve full document details including metadata, sections, and tables.",
)
async def get_document(
    doc_id: uuid.UUID,
    store: InMemoryDocumentStore = Depends(get_document_store),
    api_key: str = Depends(verify_api_key),
) -> DocumentDetailResponse:
    """Return full document details."""
    doc = store.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    return DocumentDetailResponse(
        id=doc.id,
        filename=doc.filename,
        status=doc.status,
        num_pages=doc.num_pages,
        file_size_bytes=doc.file_size_bytes,
        error_message=doc.error_message,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
        metadata=doc.metadata,
        sections=doc.sections,
        tables=doc.tables,
    )


# ------------------------------------------------------------------
# GET /{doc_id}/status
# ------------------------------------------------------------------


@router.get(
    "/{doc_id}/status",
    response_model=TaskStatusResponse,
    summary="Get processing status",
    description="Return just the processing status for a document.",
)
async def get_document_status(
    doc_id: uuid.UUID,
    store: InMemoryDocumentStore = Depends(get_document_store),
    api_key: str = Depends(verify_api_key),
) -> TaskStatusResponse:
    """Return the current processing status of a document."""
    doc = store.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    return TaskStatusResponse(
        document_id=doc.id,
        status=doc.status,
        task_id=doc.task_id,
        error_message=doc.error_message,
    )


# ------------------------------------------------------------------
# GET / (list)
# ------------------------------------------------------------------


@router.get(
    "/",
    response_model=DocumentListResponse,
    summary="List documents",
    description="List all documents with pagination and optional status filtering.",
)
async def list_documents(
    offset: int = 0,
    limit: int = 20,
    status: DocumentStatus | None = None,
    store: InMemoryDocumentStore = Depends(get_document_store),
    api_key: str = Depends(verify_api_key),
) -> DocumentListResponse:
    """Return a paginated list of documents."""
    limit = min(max(1, limit), 100)

    docs, total = store.list_all(offset=offset, limit=limit, status_filter=status)

    items = [
        DocumentListItem(
            id=d.id,
            filename=d.filename,
            status=d.status,
            num_pages=d.num_pages,
            created_at=d.created_at,
            metadata_title=d.metadata.title if d.metadata else None,
        )
        for d in docs
    ]

    return DocumentListResponse(
        documents=items,
        total=total,
        offset=offset,
        limit=limit,
    )


# ------------------------------------------------------------------
# DELETE /{doc_id}
# ------------------------------------------------------------------


@router.delete(
    "/{doc_id}",
    status_code=204,
    summary="Delete a document",
    description="Delete a document and all associated data (vectors, file).",
)
async def delete_document(
    doc_id: uuid.UUID,
    store: InMemoryDocumentStore = Depends(get_document_store),
    api_key: str = Depends(verify_api_key),
) -> None:
    """Delete a document, its file, and its vector store entries."""
    doc = store.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Delete file from disk
    file_path = Path(doc.file_path)
    if file_path.exists():
        file_path.unlink()

    # Delete from vector store (best-effort)
    try:
        from src.retrieval.vector_store import VectorStore

        VectorStore().delete_paper(str(doc_id))
    except Exception as exc:
        logger.warning("Vector store cleanup failed for %s: %s", doc_id, exc)

    store.delete(doc_id)
    logger.info("Document deleted: %s", doc_id)


# ------------------------------------------------------------------
# GET /{doc_id}/sections
# ------------------------------------------------------------------


@router.get(
    "/{doc_id}/sections",
    summary="Get document sections",
    description="Return the detected sections for a document.",
)
async def get_document_sections(
    doc_id: uuid.UUID,
    store: InMemoryDocumentStore = Depends(get_document_store),
    api_key: str = Depends(verify_api_key),
) -> dict:
    """Return sections detected during layout analysis."""
    doc = store.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    return {
        "document_id": str(doc.id),
        "sections": [s.model_dump() for s in doc.sections],
        "total": len(doc.sections),
    }


# ------------------------------------------------------------------
# GET /{doc_id}/tables
# ------------------------------------------------------------------


@router.get(
    "/{doc_id}/tables",
    summary="Get document tables",
    description="Return the extracted tables for a document.",
)
async def get_document_tables(
    doc_id: uuid.UUID,
    store: InMemoryDocumentStore = Depends(get_document_store),
    api_key: str = Depends(verify_api_key),
) -> dict:
    """Return tables extracted from the document."""
    doc = store.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    return {
        "document_id": str(doc.id),
        "tables": [t.model_dump() for t in doc.tables],
        "total": len(doc.tables),
    }
