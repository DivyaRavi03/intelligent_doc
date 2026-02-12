"""FastAPI entry point for the Intelligent Document Processing platform."""

from __future__ import annotations

import hashlib
import logging
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.ingestion.layout_analyzer import LayoutAnalyzer
from src.ingestion.metadata_extractor import MetadataExtractor
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.table_extractor import TableExtractor
from src.api.routes_extract import router as feedback_router
from src.models.schemas import (
    DocumentMetadataSchema,
    DocumentStatus,
    DocumentUploadResponse,
    HealthResponse,
    LayoutAnalysisResult,
    PDFExtractionResult,
    TableExtractionResult,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Intelligent Document Processing",
    description="AI-powered research paper analysis platform",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(feedback_router)


@app.on_event("startup")
async def startup() -> None:
    """Ensure required directories exist on startup."""
    settings.ensure_dirs()
    logger.info("Upload dir: %s", settings.upload_dir)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Liveness / readiness probe."""
    return HealthResponse()


@app.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile) -> DocumentUploadResponse:
    """Upload a PDF document for processing.

    The file is saved to disk and a document record is returned.
    Full processing (text extraction, table extraction, layout analysis,
    metadata extraction) will be triggered asynchronously in Phase 2 via Celery.
    For now, processing runs synchronously for demonstration.
    """
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

    # Save to disk
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    save_path = upload_dir / f"{doc_id}.pdf"
    save_path.write_bytes(content)

    return DocumentUploadResponse(
        id=doc_id,
        filename=file.filename,
        status=DocumentStatus.PENDING,
        created_at=save_path.stat().st_mtime,
    )


@app.post("/documents/{doc_id}/process")
async def process_document(doc_id: str) -> dict:
    """Synchronous processing endpoint (Phase 1 demo).

    Runs the full ingestion pipeline on an already-uploaded document.
    In production this would be a Celery task.
    """
    pdf_path = Path(settings.upload_dir) / f"{doc_id}.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Document not found")

    # 1. PDF text extraction
    parser = PDFParser()
    extraction: PDFExtractionResult = parser.extract(pdf_path)

    # 2. Table extraction
    table_extractor = TableExtractor()
    tables: TableExtractionResult = table_extractor.extract_tables(pdf_path)

    # 3. Layout analysis
    layout_analyzer = LayoutAnalyzer()
    layout: LayoutAnalysisResult = layout_analyzer.analyze(extraction.pages)

    # 4. Metadata extraction
    meta_extractor = MetadataExtractor()
    metadata: DocumentMetadataSchema = meta_extractor.extract(
        extraction.pages, use_llm=bool(settings.gemini_api_key)
    )

    return {
        "document_id": doc_id,
        "status": "completed",
        "total_pages": extraction.total_pages,
        "avg_confidence": extraction.avg_confidence,
        "sections_found": len(layout.sections),
        "tables_found": tables.total_tables,
        "references_found": len(layout.references),
        "metadata": metadata.model_dump(),
    }
