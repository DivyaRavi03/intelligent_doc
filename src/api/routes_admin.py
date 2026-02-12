"""Admin and monitoring endpoints.

Provides health checks, metrics, cost breakdowns, and evaluation results.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from src.api.auth import optional_api_key, verify_api_key
from src.api.rate_limiter import ADMIN_LIMIT, limiter
from src.api.stores import get_metrics
from src.models.schemas import (
    CostBreakdown,
    EvaluationResult,
    HealthCheckResponse,
    MetricsResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


# ------------------------------------------------------------------
# GET /health
# ------------------------------------------------------------------


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Service health check",
    description="Check the health of all service dependencies.",
)
async def health_check(
    api_key: str | None = Depends(optional_api_key),
) -> HealthCheckResponse:
    """Check health of database, Redis, and ChromaDB."""
    redis_ok = False
    try:
        import redis

        from src.config import settings

        r = redis.from_url(settings.redis_url, socket_connect_timeout=2)
        redis_ok = r.ping()
    except Exception:
        pass

    chromadb_ok = False
    try:
        import chromadb

        client = chromadb.Client()
        client.heartbeat()
        chromadb_ok = True
    except Exception:
        pass

    return HealthCheckResponse(
        status="healthy",
        version="0.5.0",
        database="ok",  # in-memory store always available
        redis="ok" if redis_ok else "unavailable",
        chromadb="ok" if chromadb_ok else "unavailable",
        timestamp=datetime.now(timezone.utc),
    )


# ------------------------------------------------------------------
# GET /metrics
# ------------------------------------------------------------------


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="System metrics",
    description="Return document, query, and token usage metrics.",
)
async def get_system_metrics(
    metrics: dict = Depends(get_metrics),
    api_key: str = Depends(verify_api_key),
) -> MetricsResponse:
    """Return aggregated system metrics."""
    return MetricsResponse(
        total_documents=metrics.get("total_documents", 0),
        total_queries=metrics.get("total_queries", 0),
        total_llm_tokens=metrics.get("total_llm_tokens", 0),
        total_cost_usd=metrics.get("total_cost_usd", 0.0),
    )


# ------------------------------------------------------------------
# GET /costs
# ------------------------------------------------------------------


@router.get(
    "/costs",
    response_model=CostBreakdown,
    summary="Cost breakdown",
    description="Return cost breakdown by operation type.",
)
async def get_costs(
    metrics: dict = Depends(get_metrics),
    api_key: str = Depends(verify_api_key),
) -> CostBreakdown:
    """Return cost breakdown by operation type."""
    return CostBreakdown(
        gemini_embedding_cost=metrics.get("embedding_cost", 0.0),
        gemini_generation_cost=metrics.get("generation_cost", 0.0),
        total_cost=metrics.get("total_cost_usd", 0.0),
        documents_processed=metrics.get("total_documents", 0),
        queries_answered=metrics.get("total_queries", 0),
    )


# ------------------------------------------------------------------
# GET /eval/latest
# ------------------------------------------------------------------


@router.get(
    "/eval/latest",
    response_model=EvaluationResult,
    summary="Latest evaluation",
    description="Return the latest evaluation run results.",
)
async def get_latest_evaluation(
    api_key: str = Depends(verify_api_key),
) -> EvaluationResult:
    """Return placeholder evaluation results."""
    return EvaluationResult(
        timestamp=datetime.now(timezone.utc),
        test_set_name="default",
        accuracy=0.0,
        faithfulness_score=0.0,
        avg_latency_ms=0.0,
    )
