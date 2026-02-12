"""In-memory stores for MVP.

Provides thread-safe document and API key storage using dictionaries.
The ORM models in :mod:`src.models.database` are ready for production
database integration — swap implementations via the FastAPI dependency
functions at the bottom of this module.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.models.schemas import (
    DocumentMetadataSchema,
    DocumentStatus,
    DetectedSection,
    ExtractedTable,
    PaperExtraction,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DocumentRecord:
    """In-memory document record matching the ORM shape."""

    id: uuid.UUID
    filename: str
    file_path: str
    file_hash: str
    file_size_bytes: int
    num_pages: int | None = None
    status: DocumentStatus = DocumentStatus.PENDING
    task_id: str | None = None
    error_message: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # Cached processing results
    metadata: DocumentMetadataSchema | None = None
    sections: list[DetectedSection] = field(default_factory=list)
    tables: list[ExtractedTable] = field(default_factory=list)
    extraction: PaperExtraction | None = None


@dataclass
class APIKeyRecord:
    """In-memory API key record."""

    key: str
    name: str
    rate_limit_uploads: int = 10
    rate_limit_queries: int = 100
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# In-memory implementations
# ---------------------------------------------------------------------------

class InMemoryDocumentStore:
    """Thread-safe in-memory document store."""

    def __init__(self) -> None:
        self._store: dict[uuid.UUID, DocumentRecord] = {}
        self._lock = threading.Lock()

    def get(self, doc_id: uuid.UUID) -> DocumentRecord | None:
        """Return a document record by ID, or None."""
        return self._store.get(doc_id)

    def save(self, doc: DocumentRecord) -> None:
        """Insert or update a document record."""
        with self._lock:
            doc.updated_at = datetime.now(timezone.utc)
            self._store[doc.id] = doc

    def delete(self, doc_id: uuid.UUID) -> bool:
        """Delete a document record. Returns True if it existed."""
        with self._lock:
            if doc_id in self._store:
                del self._store[doc_id]
                return True
            return False

    def list_all(
        self,
        offset: int = 0,
        limit: int = 20,
        status_filter: DocumentStatus | None = None,
    ) -> tuple[list[DocumentRecord], int]:
        """Return paginated documents sorted by created_at descending."""
        docs = sorted(self._store.values(), key=lambda d: d.created_at, reverse=True)
        if status_filter is not None:
            docs = [d for d in docs if d.status == status_filter]
        total = len(docs)
        return docs[offset : offset + limit], total

    def update_status(
        self,
        doc_id: uuid.UUID,
        status: DocumentStatus,
        error: str | None = None,
    ) -> None:
        """Update the processing status of a document."""
        with self._lock:
            doc = self._store.get(doc_id)
            if doc:
                doc.status = status
                doc.error_message = error
                doc.updated_at = datetime.now(timezone.utc)

    def count(self) -> int:
        """Return total number of documents."""
        return len(self._store)


class InMemoryAPIKeyStore:
    """Thread-safe API key store with a pre-populated test key."""

    def __init__(self) -> None:
        self._keys: dict[str, APIKeyRecord] = {
            "test-api-key-12345": APIKeyRecord(
                key="test-api-key-12345",
                name="Test Key",
            ),
        }

    def validate(self, key: str) -> bool:
        """Return True if the key exists."""
        return key in self._keys

    def get_limits(self, key: str) -> dict[str, Any]:
        """Return rate limits for a key."""
        record = self._keys.get(key)
        if not record:
            return {}
        return {
            "uploads_per_hour": record.rate_limit_uploads,
            "queries_per_hour": record.rate_limit_queries,
        }


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

_doc_store = InMemoryDocumentStore()
_key_store = InMemoryAPIKeyStore()
_metrics: dict[str, Any] = {
    "total_documents": 0,
    "total_queries": 0,
    "total_llm_tokens": 0,
    "total_cost_usd": 0.0,
    "total_processing_time": 0.0,
}
_metrics_lock = threading.Lock()


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------

def get_document_store() -> InMemoryDocumentStore:
    """FastAPI dependency for document store."""
    return _doc_store


def get_api_key_store() -> InMemoryAPIKeyStore:
    """FastAPI dependency for API key store."""
    return _key_store


def get_metrics() -> dict[str, Any]:
    """FastAPI dependency for metrics dictionary."""
    return _metrics


def update_metrics(key: str, value: float) -> None:
    """Thread-safe increment of a metrics counter."""
    with _metrics_lock:
        _metrics[key] = _metrics.get(key, 0.0) + value
