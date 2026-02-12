"""FastAPI entry point for the Intelligent Document Processing platform.

Registers all API routers, configures middleware, rate limiting, and
provides a WebSocket endpoint for real-time processing updates.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from src.api.rate_limiter import limiter
from src.api.routes_admin import router as admin_router
from src.api.routes_documents import router as documents_router
from src.api.routes_extract import router as extract_router
from src.api.routes_query import router as query_router
from src.api.websocket import processing_websocket
from src.config import settings
from src.models.schemas import HealthResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    """Application lifespan: startup and shutdown."""
    settings.ensure_dirs()
    logger.info("Upload dir: %s", settings.upload_dir)
    yield


app = FastAPI(
    title="Intelligent Document Processing",
    description="AI-powered research paper analysis platform",
    version="0.5.0",
    lifespan=lifespan,
)

# -- Middleware --
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -- Rate limiter --
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# -- Routers --
app.include_router(documents_router)
app.include_router(query_router)
app.include_router(extract_router)
app.include_router(admin_router)


# -- WebSocket --
@app.websocket("/ws/processing/{task_id}")
async def ws_processing(websocket, task_id: str):  # type: ignore[no-untyped-def]
    """WebSocket endpoint for real-time processing status."""
    await processing_websocket(websocket, task_id)


# -- Root health check --
@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Liveness / readiness probe."""
    return HealthResponse()
