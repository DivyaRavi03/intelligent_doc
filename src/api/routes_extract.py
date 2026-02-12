"""Feedback endpoint for extraction corrections.

Provides a ``POST /api/v1/feedback`` route that stores user corrections
to extracted paper data.  An in-memory store is used for the MVP; the
:class:`~src.models.database.ExtractionFeedback` ORM model is ready for
database integration.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter

from src.models.schemas import FeedbackRequest, FeedbackResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["feedback"])

_FEEDBACK_STORE: list[dict] = []
_FEEDBACK_THRESHOLD = 5


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
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
