"""Extraction and feedback endpoints.

Provides extraction retrieval, re-extraction, feedback submission, and
feedback statistics.  Builds on the Phase 4 feedback store with added
authentication and new endpoints.
"""

from __future__ import annotations

import logging
import uuid
from collections import Counter
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from src.api.auth import verify_api_key
from src.api.stores import (
    InMemoryDocumentStore,
    get_document_store,
)
from src.models.schemas import (
    DocumentStatus,
    ExtractionStatsResponse,
    FeedbackRequest,
    FeedbackResponse,
    ReExtractRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["extraction"])

_FEEDBACK_STORE: list[dict] = []
_FEEDBACK_THRESHOLD = 5


# ------------------------------------------------------------------
# POST /feedback
# ------------------------------------------------------------------


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    api_key: str = Depends(verify_api_key),
) -> FeedbackResponse:
    """Record a user correction to an extracted field.

    When the number of corrections for a given ``(paper_id, field_name)``
    pair exceeds :data:`_FEEDBACK_THRESHOLD`, a warning is logged to
    flag the extraction prompt for review.
    """
    feedback_id = uuid.uuid4()
    now = datetime.now(timezone.utc)

    entry = {
        "id": feedback_id,
        "paper_id": request.paper_id,
        "field_name": request.field_name,
        "original_value": request.original_value,
        "corrected_value": request.corrected_value,
        "user_comment": request.user_comment,
        "created_at": now,
    }
    _FEEDBACK_STORE.append(entry)

    # Check threshold
    count = sum(
        1
        for f in _FEEDBACK_STORE
        if f["paper_id"] == request.paper_id
        and f["field_name"] == request.field_name
    )
    if count >= _FEEDBACK_THRESHOLD:
        logger.warning(
            "Feedback threshold reached for paper_id=%s field=%s (%d corrections)",
            request.paper_id,
            request.field_name,
            count,
        )

    return FeedbackResponse(
        id=feedback_id,
        paper_id=request.paper_id,
        field_name=request.field_name,
        created_at=now,
    )


# ------------------------------------------------------------------
# GET /documents/{doc_id}/extraction
# ------------------------------------------------------------------


@router.get(
    "/documents/{doc_id}/extraction",
    summary="Get extraction results",
    description="Return the structured extraction (PaperExtraction) for a document.",
)
async def get_extraction(
    doc_id: uuid.UUID,
    store: InMemoryDocumentStore = Depends(get_document_store),
    api_key: str = Depends(verify_api_key),
) -> dict:
    """Return cached PaperExtraction for a processed document."""
    doc = store.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    if doc.extraction is None:
        raise HTTPException(
            status_code=404, detail="No extraction results available"
        )

    return {
        "document_id": str(doc.id),
        "extraction": doc.extraction.model_dump(),
    }


# ------------------------------------------------------------------
# POST /documents/{doc_id}/re-extract
# ------------------------------------------------------------------


@router.post(
    "/documents/{doc_id}/re-extract",
    summary="Re-run extraction",
    description="Re-run the structured extraction pipeline on a processed document.",
)
async def re_extract(
    doc_id: uuid.UUID,
    body: ReExtractRequest,
    store: InMemoryDocumentStore = Depends(get_document_store),
    api_key: str = Depends(verify_api_key),
) -> dict:
    """Re-run PaperExtractor on a completed document."""
    doc = store.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    if doc.status != DocumentStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Document not completed (status: {doc.status.value})",
        )

    paper_text = "\n\n".join(s.text for s in doc.sections) if doc.sections else ""
    if not paper_text.strip():
        raise HTTPException(status_code=400, detail="Document has no text content")

    try:
        from src.llm.extractor import PaperExtractor
        from src.llm.gemini_client import GeminiClient

        client = GeminiClient()
        extractor = PaperExtractor(client)
        extraction = extractor.extract(paper_text, doc.metadata)

        doc.extraction = extraction
        store.save(doc)
    except Exception as exc:
        logger.exception("Re-extraction failed for %s", doc_id)
        raise HTTPException(
            status_code=500, detail=f"Re-extraction failed: {exc}"
        ) from exc

    return {
        "document_id": str(doc.id),
        "status": "completed",
        "extraction": extraction.model_dump(),
    }


# ------------------------------------------------------------------
# GET /feedback/stats
# ------------------------------------------------------------------


@router.get(
    "/feedback/stats",
    response_model=ExtractionStatsResponse,
    summary="Feedback statistics",
    description="Return aggregated feedback statistics.",
)
async def feedback_stats(
    api_key: str = Depends(verify_api_key),
) -> ExtractionStatsResponse:
    """Return feedback counts by field and flagged papers."""
    total = len(_FEEDBACK_STORE)

    by_field: dict[str, int] = dict(
        Counter(f["field_name"] for f in _FEEDBACK_STORE)
    )

    # Papers with >= threshold corrections on any single field
    paper_field_counts: dict[tuple[str, str], int] = Counter(
        (f["paper_id"], f["field_name"]) for f in _FEEDBACK_STORE
    )
    flagged = list(
        {pf[0] for pf, count in paper_field_counts.items() if count >= _FEEDBACK_THRESHOLD}
    )

    return ExtractionStatsResponse(
        total_feedback=total,
        by_field=by_field,
        flagged_papers=flagged,
    )
