"""API key authentication for FastAPI endpoints.

Provides a ``verify_api_key`` dependency that validates the ``X-API-Key``
header against the :class:`~src.api.stores.InMemoryAPIKeyStore`.
"""

from __future__ import annotations

import logging

from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

from src.api.stores import InMemoryAPIKeyStore, get_api_key_store

logger = logging.getLogger(__name__)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key: str | None = Security(api_key_header),
    key_store: InMemoryAPIKeyStore = Depends(get_api_key_store),
) -> str:
    """Validate the X-API-Key header.

    Returns the validated key string on success.

    Raises:
        HTTPException: 401 if key is missing or invalid.
    """
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing X-API-Key header",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if not key_store.validate(api_key):
        logger.warning("Invalid API key: %s...", api_key[:8])
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return api_key


async def optional_api_key(
    api_key: str | None = Security(api_key_header),
) -> str | None:
    """Extract API key without validation (for public endpoints)."""
    return api_key
