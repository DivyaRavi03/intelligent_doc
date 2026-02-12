"""Integration tests for the document processing pipeline.

These tests call ``_process_document_impl`` directly — no Celery needed.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from src.api.stores import (
    DocumentRecord,
    InMemoryDocumentStore,
    get_document_store,
)
from src.models.schemas import DocumentStatus
from src.workers.tasks import _process_document_impl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_doc_in_store(doc_id: uuid.UUID, pdf_path: Path) -> InMemoryDocumentStore:
    """Create a store with a pending document record."""
    store = get_document_store()
    doc = DocumentRecord(
        id=doc_id,
        filename="test.pdf",
        file_path=str(pdf_path),
        file_hash="integration-test",
        file_size_bytes=pdf_path.stat().st_size,
        status=DocumentStatus.PENDING,
    )
    store.save(doc)
    return store


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """Integration tests for the full processing pipeline."""

    def test_pipeline_completes(self, tmp_pdf: Path) -> None:
        """Pipeline completes successfully on a real PDF."""
        doc_id = uuid.uuid4()
        store = _setup_doc_in_store(doc_id, tmp_pdf)

        result = _process_document_impl(str(doc_id), tmp_pdf)

        assert result["status"] == "completed"
        assert result["total_pages"] >= 1
        assert result["sections_found"] >= 1
        assert result["chunks_created"] >= 1
        assert "processing_time_sec" in result

        # Verify store was updated
        doc = store.get(doc_id)
        assert doc is not None
        assert doc.status == DocumentStatus.COMPLETED

    def test_pipeline_stores_results(self, tmp_pdf: Path) -> None:
        """Pipeline saves metadata, sections, and tables to store."""
        doc_id = uuid.uuid4()
        store = _setup_doc_in_store(doc_id, tmp_pdf)

        _process_document_impl(str(doc_id), tmp_pdf)

        doc = store.get(doc_id)
        assert doc is not None
        assert doc.num_pages is not None and doc.num_pages >= 1
        assert doc.metadata is not None
        assert len(doc.sections) >= 1

    def test_pipeline_invalid_pdf(self, tmp_path: Path) -> None:
        """Pipeline marks FAILED status on invalid PDF."""
        doc_id = uuid.uuid4()
        bad_pdf = tmp_path / f"{doc_id}.pdf"
        bad_pdf.write_bytes(b"this is not a pdf")
        store = _setup_doc_in_store(doc_id, bad_pdf)

        with pytest.raises(Exception):
            _process_document_impl(str(doc_id), bad_pdf)

        doc = store.get(doc_id)
        assert doc is not None
        assert doc.status == DocumentStatus.FAILED

    def test_pipeline_progress_callback(self, tmp_pdf: Path) -> None:
        """Progress callback is invoked during processing."""
        doc_id = uuid.uuid4()
        _setup_doc_in_store(doc_id, tmp_pdf)

        stages_seen: list[str] = []

        def track_progress(stage: str, progress: float, step: str) -> None:
            stages_seen.append(stage)

        _process_document_impl(str(doc_id), tmp_pdf, update_fn=track_progress)

        assert len(stages_seen) >= 3
        assert "pdf_extraction" in stages_seen
