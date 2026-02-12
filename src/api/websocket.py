"""WebSocket endpoint for real-time processing status updates.

Polls the Celery AsyncResult for a given task and pushes status
updates to the connected client every second until the task reaches
a terminal state.
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import WebSocket, WebSocketDisconnect

from src.models.schemas import ProcessingUpdate

logger = logging.getLogger(__name__)


async def processing_websocket(websocket: WebSocket, task_id: str) -> None:
    """Stream processing status updates for a Celery task.

    Accepts the WebSocket connection, then polls the task result every
    second.  Sends :class:`ProcessingUpdate` JSON messages until the
    task reaches a terminal state (SUCCESS, FAILURE, REVOKED) or the
    client disconnects.
    """
    await websocket.accept()

    try:
        from celery.result import AsyncResult

        from src.workers.celery_app import celery_app

        result = AsyncResult(task_id, app=celery_app)
    except Exception as exc:
        logger.warning("Celery unavailable for WebSocket: %s", exc)
        update = ProcessingUpdate(
            task_id=task_id,
            status="error",
            error=f"Celery unavailable: {exc}",
        )
        await websocket.send_text(update.model_dump_json())
        await websocket.close()
        return

    try:
        while True:
            state = result.state
            meta = result.info if isinstance(result.info, dict) else {}

            if state == "PROCESSING":
                update = ProcessingUpdate(
                    task_id=task_id,
                    status="processing",
                    stage=meta.get("stage"),
                    progress=meta.get("progress", 0.0),
                    step=meta.get("step"),
                )
            elif state == "SUCCESS":
                update = ProcessingUpdate(
                    task_id=task_id,
                    status="completed",
                    progress=1.0,
                    result=result.result,
                )
                await websocket.send_text(update.model_dump_json())
                break
            elif state in ("FAILURE", "REVOKED"):
                update = ProcessingUpdate(
                    task_id=task_id,
                    status="failed",
                    error=str(result.info),
                )
                await websocket.send_text(update.model_dump_json())
                break
            elif state == "PENDING":
                update = ProcessingUpdate(
                    task_id=task_id,
                    status="pending",
                    progress=0.0,
                )
            else:
                update = ProcessingUpdate(
                    task_id=task_id,
                    status=state.lower(),
                    progress=meta.get("progress", 0.0),
                )

            await websocket.send_text(update.model_dump_json())
            await asyncio.sleep(1)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for task %s", task_id)
    except Exception as exc:
        logger.exception("WebSocket error for task %s", task_id)
        try:
            error_update = ProcessingUpdate(
                task_id=task_id,
                status="error",
                error=str(exc),
            )
            await websocket.send_text(error_update.model_dump_json())
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
