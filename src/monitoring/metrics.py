"""Prometheus metrics for document processing, retrieval, and evaluation.

Complements the LLM-specific metrics defined in
:mod:`src.llm.gemini_client`.  All metrics are lazily initialised to
avoid import-time errors when ``prometheus_client`` is unavailable.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_INITIALISED = False

# Placeholders -- populated by _init_metrics()
document_processing_duration_seconds: Any = None
document_processing_total: Any = None
retrieval_latency_seconds: Any = None
llm_request_duration_seconds: Any = None
llm_tokens_total: Any = None
llm_cost_dollars: Any = None
retrieval_precision: Any = None
extraction_accuracy: Any = None


def _init_metrics() -> None:
    """Lazily initialise Prometheus metrics (once)."""
    global _INITIALISED  # noqa: PLW0603
    global document_processing_duration_seconds, document_processing_total
    global retrieval_latency_seconds, llm_request_duration_seconds
    global llm_tokens_total, llm_cost_dollars
    global retrieval_precision, extraction_accuracy
    if _INITIALISED:
        return
    try:
        from prometheus_client import Counter, Gauge, Histogram

        document_processing_duration_seconds = Histogram(
            "document_processing_duration_seconds",
            "Duration of the full document processing pipeline",
            ["status"],
        )
        document_processing_total = Counter(
            "document_processing_total",
            "Total documents processed",
            ["status"],
        )
        retrieval_latency_seconds = Histogram(
            "retrieval_latency_seconds",
            "Latency of retrieval queries",
        )
        llm_request_duration_seconds = Histogram(
            "llm_request_duration_seconds",
            "Duration of individual LLM requests",
            ["model", "endpoint"],
        )
        llm_tokens_total = Counter(
            "llm_tokens_total",
            "Total LLM tokens by model and type",
            ["model", "token_type"],
        )
        llm_cost_dollars = Counter(
            "llm_cost_dollars",
            "Cumulative LLM cost in USD",
        )
        retrieval_precision = Gauge(
            "retrieval_precision",
            "Latest retrieval precision@k from evaluation",
        )
        extraction_accuracy = Gauge(
            "extraction_accuracy",
            "Latest extraction accuracy from evaluation",
        )
    except ImportError:
        logger.debug("prometheus_client not installed, metrics disabled")
    _INITIALISED = True


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def record_document_processing(duration_sec: float, status: str) -> None:
    """Record a document processing event."""
    _init_metrics()
    if document_processing_duration_seconds is not None:
        document_processing_duration_seconds.labels(status=status).observe(duration_sec)
    if document_processing_total is not None:
        document_processing_total.labels(status=status).inc()


def record_retrieval_latency(duration_sec: float) -> None:
    """Record a retrieval query latency."""
    _init_metrics()
    if retrieval_latency_seconds is not None:
        retrieval_latency_seconds.observe(duration_sec)


def record_llm_request(
    duration_sec: float,
    model: str,
    endpoint: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
) -> None:
    """Record a single LLM request with timing, tokens, and cost."""
    _init_metrics()
    if llm_request_duration_seconds is not None:
        llm_request_duration_seconds.labels(model=model, endpoint=endpoint).observe(
            duration_sec
        )
    if llm_tokens_total is not None:
        llm_tokens_total.labels(model=model, token_type="input").inc(input_tokens)
        llm_tokens_total.labels(model=model, token_type="output").inc(output_tokens)
    if llm_cost_dollars is not None:
        llm_cost_dollars.inc(cost_usd)


def update_eval_gauges(precision: float, accuracy: float) -> None:
    """Update evaluation gauge metrics after a benchmark run."""
    _init_metrics()
    if retrieval_precision is not None:
        retrieval_precision.set(precision)
    if extraction_accuracy is not None:
        extraction_accuracy.set(accuracy)
