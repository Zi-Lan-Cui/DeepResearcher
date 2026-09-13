"""Authenticated SSE replay and live-tail route."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from deepresearcher.service.events.projector import project
from deepresearcher.service.events.stream import CLOSE_STREAM
from deepresearcher.service.persistence.models import Run
from deepresearcher.service.web.dependencies import app_state, owned_run

router = APIRouter(prefix="/api/runs")

SSE_HEARTBEAT_SECONDS = 15.0
SSE_DB_POLL_SECONDS = 1.0
logger = logging.getLogger("deepresearcher.service.web.routes.events")


def _sse(frame: Any) -> str:
    data = json.dumps(frame.data, ensure_ascii=False, default=str)
    return f"id: {frame.data.get('seq', 0)}\nevent: {frame.event}\ndata: {data}\n\n"


@router.get("/{run_id}/events")
async def run_events(
    request: Request,
    run: Run = Depends(owned_run),
) -> StreamingResponse:
    state = app_state(request)

    async def stream() -> AsyncIterator[str]:
        key, queue = state.fanout.subscribe(run.id)
        notify_key, notify_queue = state.manager.signal_bus.subscribe_event(run.id)
        preview_subscription = None
        if state.ephemeral_bus is not None:
            try:
                preview_subscription = await state.ephemeral_bus.subscribe(run.id)
            except Exception:  # noqa: BLE001 - durable DB stream remains available
                logger.warning("redis_preview_subscribe_failed run_id=%s", run.id, exc_info=True)
        last_seq = 0
        last_ping = asyncio.get_running_loop().time()
        try:
            while True:
                rows = await state.manager.event_store.after(run.id, last_seq)
                for row in rows:
                    last_seq = row.seq
                    frame = project(row.record)
                    if frame is not None:
                        yield _sse(frame)
                        if frame.event == "done":
                            return
                try:
                    local_wait = asyncio.create_task(queue.get())
                    notify_wait = asyncio.create_task(notify_queue.get())
                    waits = [local_wait, notify_wait]
                    preview_wait = None
                    if preview_subscription is not None:
                        preview_wait = asyncio.create_task(preview_subscription.queue.get())
                        waits.append(preview_wait)
                    done, pending = await asyncio.wait(
                        waits,
                        timeout=SSE_DB_POLL_SECONDS,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for waiter in pending:
                        waiter.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
                    if not done:
                        raise TimeoutError
                    items = []
                    if local_wait in done:
                        items.append(local_wait.result())
                    if preview_wait is not None and preview_wait in done:
                        items.append(preview_wait.result())
                    if notify_wait in done:
                        notify_wait.result()
                except TimeoutError:
                    now = asyncio.get_running_loop().time()
                    if now - last_ping >= SSE_HEARTBEAT_SECONDS:
                        last_ping = now
                        yield ": ping\n\n"
                    continue
                for item in items:
                    if item is CLOSE_STREAM or item is None:
                        continue
                    if item.get("event_type") == "text_delta":
                        frame = project(item)
                        if frame is not None:
                            yield _sse(frame)
        finally:
            state.fanout.unsubscribe(run.id, key)
            state.manager.signal_bus.unsubscribe("event_committed", notify_key)
            if preview_subscription is not None:
                await preview_subscription.close()

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
