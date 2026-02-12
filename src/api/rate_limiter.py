"""Rate limiting for API endpoints using slowapi.

Provides a pre-configured :class:`slowapi.Limiter` and rate limit constants
for upload, query, and admin endpoints.
"""

from __future__ import annotations

import logging

from fastapi import Request, Response
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

logger = logging.getLogger(__name__)

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200/hour"],
    storage_uri="memory://",
)

# Rate limit strings
UPLOAD_LIMIT = "10/hour"
QUERY_LIMIT = "100/hour"
ADMIN_LIMIT = "20/minute"


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> Response:
    """Return 429 with Retry-After header when rate limit is exceeded."""
    logger.warning(
        "Rate limit exceeded for %s on %s",
        get_remote_address(request),
        request.url.path,
    )
    retry_after = str(getattr(exc, "retry_after", 60) or 60)
    return Response(
        content=f'{{"detail":"Rate limit exceeded: {exc.detail}"}}',
        status_code=429,
        media_type="application/json",
        headers={"Retry-After": retry_after},
    )
