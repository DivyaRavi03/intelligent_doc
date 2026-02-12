"""Celery application configuration.

Uses Redis as broker and result backend.  Start a worker with::

    celery -A src.workers.celery_app worker --loglevel=info
"""

from __future__ import annotations

import logging

from celery import Celery

from src.config import settings

logger = logging.getLogger(__name__)

celery_app = Celery(
    "intelligent_doc",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["src.workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    result_expires=3600,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)
