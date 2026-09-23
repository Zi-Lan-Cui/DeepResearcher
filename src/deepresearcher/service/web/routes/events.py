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
        # 两条唤醒源、一条数据路:已提交帧只从 DB tail 读(notify 或 poll 叫醒);
        # 预览帧读 EphemeralEventBus 订阅。没有进程内直投,也就不需要去重。
        notify_key, notify_queue = state.manager.signal_bus.subscribe_event(run.id)
        preview_subscription = None
        if state.ephemeral_bus is not None:
            try:
                preview_subscription = await state.ephemeral_bus.subscribe(run.id)
            except Exception:  # noqa: BLE001 - durable DB stream remains available
                logger.warning("preview_subscribe_failed run_id=%s", run.id, exc_info=True)
        last_seq = 0
        last_ping = asyncio.get_running_loop().time()

        def _drain(rows) -> tuple[list[str], bool]:
            """把 DB 行投成 SSE 帧;返回 (frames, 是否见到 done)。"""
            nonlocal last_seq
            frames: list[str] = []
            saw_done = False
            for row in rows:
                last_seq = row.seq
                frame = project(row.record)
                if frame is not None:
                    frames.append(_sse(frame))
                    saw_done = saw_done or frame.event == "done"
            return frames, saw_done

        async def _terminate_if_settled() -> tuple[list[str], bool]:
            """done 帧可能整批丢在失败的 flush 里——行的终态才是权威。

            重排一次 tail,防止合成帧排在刚落库的 done 之前;仍没有才合成。
            """
            frames, saw_done = _drain(await state.manager.event_store.after(run.id, last_seq))
            if saw_done:
                return frames, True
            terminal = await state.manager.terminal_state(run.id)
            if terminal is None:
                return frames, False
            synthesized = project(
                {
                    "run_id": run.id,
                    "event_type": "run_done",
                    "seq": last_seq,
                    "payload": terminal,
                }
            )
            if synthesized is not None:
                frames.append(_sse(synthesized))
            return frames, True

        try:
            while True:
                rows = await state.manager.event_store.after(run.id, last_seq)
                frames, saw_done = _drain(rows)
                for frame in frames:
                    yield frame
                if saw_done:
                    state.hub.close(run.id)
                    return
                if not rows:
                    frames, settled = await _terminate_if_settled()
                    for frame in frames:
                        yield frame
                    if settled:
                        state.hub.close(run.id)
                        return
                notify_wait = asyncio.create_task(notify_queue.get())
                waits = [notify_wait]
                preview_wait = None
                if preview_subscription is not None:
                    preview_wait = asyncio.create_task(preview_subscription.queue.get())
                    waits.append(preview_wait)
                try:
                    done, _ = await asyncio.wait(
                        waits,
                        timeout=SSE_DB_POLL_SECONDS,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                finally:
                    # 成功路径只回收未完成的挂起者;客户端断连时 asyncio.wait 不为
                    # 子任务负责——两条退出路径都在这里回收挂起任务。
                    for waiter in waits:
                        waiter.cancel()
                    await asyncio.gather(*waits, return_exceptions=True)
                items: list[Any] = []
                if preview_wait is not None and preview_wait in done:
                    items.append(preview_wait.result())
                if notify_wait in done:
                    # notify 只是叫醒:payload 是 None,数据由下一轮循环头的 tail 取。
                    notify_wait.result()
                if not items:
                    # 本轮无预览可发(wait 超时或仅 notify 叫醒):
                    # 走一次心跳检查再回轮询头。超时是常态,不是异常——绝不许上抛。
                    now = asyncio.get_running_loop().time()
                    if now - last_ping >= SSE_HEARTBEAT_SECONDS:
                        last_ping = now
                        yield ": ping\n\n"
                    continue
                for item in items:
                    if item.get("event_type") == "text_delta":
                        frame = project(item)
                        if frame is not None:
                            yield _sse(frame)
        finally:
            state.manager.signal_bus.unsubscribe("event_committed", notify_key)
            if preview_subscription is not None:
                await preview_subscription.close()

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
