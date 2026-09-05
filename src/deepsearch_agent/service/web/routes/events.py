"""Authenticated SSE replay and live-tail route."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from deepsearch_agent.service.events import CLOSE_STREAM
from deepsearch_agent.service.models import Run
from deepsearch_agent.service.projector import project
from deepsearch_agent.service.web.dependencies import app_state, owned_run

router = APIRouter(prefix="/api/runs")

SSE_HEARTBEAT_SECONDS = 15.0
SSE_DB_POLL_SECONDS = 1.0


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
        notify_key, notify_queue = state.manager.event_notifier.subscribe(run.id)
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
                    done, pending = await asyncio.wait(
                        (local_wait, notify_wait),
                        timeout=SSE_DB_POLL_SECONDS,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for waiter in pending:
                        waiter.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
                    if not done:
                        raise TimeoutError
                    item = local_wait.result() if local_wait in done else None
                    for completed in done:
                        if completed is not local_wait:
                            completed.result()
                except TimeoutError:
                    now = asyncio.get_running_loop().time()
                    if now - last_ping >= SSE_HEARTBEAT_SECONDS:
                        last_ping = now
                        yield ": ping\n\n"
                    continue
                if item is CLOSE_STREAM or item is None:
                    continue
                if item.get("event_type") == "text_delta":
                    frame = project(item)
                    if frame is not None:
                        yield _sse(frame)
        finally:
            state.fanout.unsubscribe(run.id, key)
            state.manager.event_notifier.unsubscribe(run.id, notify_key)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
