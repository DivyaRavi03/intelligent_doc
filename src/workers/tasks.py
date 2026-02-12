"""Celery tasks for asynchronous document processing.

The heavy lifting lives in :func:`_process_document_impl`, a plain function
that is fully testable without Celery infrastructure.  The
:func:`process_document` task is a thin Celery wrapper that delegates to it.
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from src.config import settings
from src.models.schemas import DocumentStatus, PaperStructure

logger = logging.getLogger(__name__)

# Stage weights for progress reporting
_STAGES: list[tuple[str, float, str]] = [
    ("pdf_extraction", 0.10, "Extracting text from PDF"),
    ("table_extraction", 0.20, "Extracting tables"),
    ("layout_analysis", 0.30, "Analyzing document layout"),
    ("metadata_extraction", 0.40, "Extracting metadata"),
    ("chunking", 0.55, "Creating semantic chunks"),
    ("embedding", 0.65, "Generating embeddings"),
    ("vector_store", 0.75, "Indexing in vector database"),
    ("extraction", 0.85, "Extracting structured data"),
    ("finalizing", 0.95, "Saving results"),
]


def _process_document_impl(
    doc_id: str,
    pdf_path: Path,
    update_fn: Callable[[str, float, str], None] | None = None,
) -> dict[str, Any]:
    """Run the full ingestion-to-extraction pipeline.

    This function is intentionally kept free of Celery dependencies so it
    can be called directly in tests and in a synchronous fallback path.

    Args:
        doc_id: Document UUID as string.
        pdf_path: Path to the uploaded PDF file.
        update_fn: Optional callback ``(stage, progress, step)`` for status
            updates (used by the Celery task wrapper).

    Returns:
        Summary dict with ``status``, ``total_pages``, ``sections_found``,
        ``tables_found``, ``chunks_created``, ``processing_time_sec``,
        and ``warnings``.
    """
    from src.api.stores import get_document_store, update_metrics
    from src.chunking.chunker import SectionAwareChunker
    from src.ingestion.layout_analyzer import LayoutAnalyzer
    from src.ingestion.metadata_extractor import MetadataExtractor
    from src.ingestion.pdf_parser import PDFParser
    from src.ingestion.table_extractor import TableExtractor

    store = get_document_store()
    start_time = time.time()
    warnings: list[str] = []
    doc_uuid = uuid.UUID(doc_id)

    def _update(stage: str, progress: float, step: str) -> None:
        if update_fn:
            update_fn(stage, progress, step)

    try:
        store.update_status(doc_uuid, DocumentStatus.PROCESSING)

        # Stage 1 — PDF text extraction
        _update("pdf_extraction", 0.10, "Extracting text from PDF")
        parser = PDFParser()
        extraction = parser.extract(pdf_path)
        logger.info("Extracted %d pages from %s", extraction.total_pages, doc_id)

        # Stage 2 — Table extraction
        _update("table_extraction", 0.20, "Extracting tables")
        table_extractor = TableExtractor()
        tables = table_extractor.extract_tables(pdf_path)
        logger.info("Extracted %d tables from %s", tables.total_tables, doc_id)

        # Stage 3 — Layout analysis
        _update("layout_analysis", 0.30, "Analyzing document layout")
        layout_analyzer = LayoutAnalyzer()
        layout = layout_analyzer.analyze(extraction.pages)
        logger.info("Detected %d sections in %s", len(layout.sections), doc_id)

        # Stage 4 — Metadata extraction
        _update("metadata_extraction", 0.40, "Extracting metadata")
        meta_extractor = MetadataExtractor()
        metadata = meta_extractor.extract(
            extraction.pages,
            use_llm=bool(settings.gemini_api_key),
        )

        # Stage 5 — Chunking
        _update("chunking", 0.55, "Creating semantic chunks")
        paper = PaperStructure(
            paper_id=doc_id,
            sections=layout.sections,
            tables=tables.tables,
            references=layout.references,
            metadata=metadata,
        )
        chunker = SectionAwareChunker(
            target_tokens=settings.chunk_size,
            overlap_tokens=settings.chunk_overlap,
        )
        chunks = chunker.chunk(paper)
        logger.info("Created %d chunks for %s", len(chunks), doc_id)

        # Stage 6 — Embeddings (skip if no API key)
        embeddings: list[list[float]] = []
        if settings.gemini_api_key and chunks:
            _update("embedding", 0.65, "Generating embeddings")
            try:
                from src.retrieval.embedding_service import EmbeddingService

                embedding_service = EmbeddingService()
                embeddings = embedding_service.embed_chunks(chunks)
                logger.info("Generated %d embeddings for %s", len(embeddings), doc_id)
            except Exception as exc:
                logger.warning("Embedding failed for %s: %s", doc_id, exc)
                warnings.append(f"Embedding generation failed: {exc}")
        else:
            warnings.append("Skipped embedding (no API key or no chunks)")

        # Stage 7 — Vector store (skip if no embeddings)
        if embeddings:
            _update("vector_store", 0.75, "Indexing in vector database")
            try:
                from src.retrieval.vector_store import VectorStore

                vector_store = VectorStore()
                vector_store.store(chunks, embeddings)
            except Exception as exc:
                logger.warning("Vector store failed for %s: %s", doc_id, exc)
                warnings.append(f"Vector store indexing failed: {exc}")

        # Stage 8 — Structured extraction (optional, non-fatal)
        paper_extraction = None
        if settings.gemini_api_key:
            _update("extraction", 0.85, "Extracting structured data")
            try:
                from src.llm.extractor import PaperExtractor
                from src.llm.gemini_client import GeminiClient

                client = GeminiClient()
                extractor = PaperExtractor(client)
                paper_text = "\n\n".join(s.text for s in layout.sections)
                paper_extraction = extractor.extract(paper_text, metadata)
            except Exception as exc:
                logger.warning("Extraction failed for %s: %s", doc_id, exc)
                warnings.append(f"Structured extraction failed: {exc}")
        else:
            warnings.append("Skipped extraction (no API key)")

        # Stage 9 — Finalize
        _update("finalizing", 0.95, "Saving results")
        doc = store.get(doc_uuid)
        if doc:
            doc.num_pages = extraction.total_pages
            doc.metadata = metadata
            doc.sections = layout.sections
            doc.tables = tables.tables
            doc.extraction = paper_extraction
            doc.status = DocumentStatus.COMPLETED
            store.save(doc)

        elapsed = time.time() - start_time
        update_metrics("total_processing_time", elapsed)

        result: dict[str, Any] = {
            "document_id": doc_id,
            "status": "completed",
            "total_pages": extraction.total_pages,
            "sections_found": len(layout.sections),
            "tables_found": tables.total_tables,
            "chunks_created": len(chunks),
            "processing_time_sec": round(elapsed, 2),
            "warnings": warnings,
        }
        logger.info("Processing completed for %s in %.2fs", doc_id, elapsed)
        return result

    except Exception:
        logger.exception("Processing failed for %s", doc_id)
        try:
            store.update_status(doc_uuid, DocumentStatus.FAILED, error=str(doc_id))
        except Exception:
            pass
        raise


def _create_celery_task():  # type: ignore[no-untyped-def]
    """Create the Celery shared_task at import time if Celery is available."""
    try:
        from celery import shared_task

        @shared_task(bind=True, name="process_document")
        def process_document(self, doc_id: str) -> dict:  # type: ignore[no-untyped-def]
            """Celery task wrapper for document processing."""
            pdf_path = Path(settings.upload_dir) / f"{doc_id}.pdf"
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF not found: {pdf_path}")

            def _celery_update(stage: str, progress: float, step: str) -> None:
                self.update_state(
                    state="PROCESSING",
                    meta={"stage": stage, "progress": progress, "step": step},
                )

            return _process_document_impl(doc_id, pdf_path, update_fn=_celery_update)

        return process_document
    except Exception:
        return None


process_document = _create_celery_task()
