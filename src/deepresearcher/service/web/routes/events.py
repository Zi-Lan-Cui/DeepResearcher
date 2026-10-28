"""带鉴权的 SSE 回放与实时尾随路由。"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from deepresearcher.observability.events.names import EventName
from deepresearcher.service.storage.tables import Run
from deepresearcher.service.web.dependencies import app_state, owned_run
from deepresearcher.service.web.projector import project

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
    """以 SSE 回放并实时尾随某个 run 的持久事件流。

    返回：
        StreamingResponse: 帧流，直到 done 或行终态为止。
    """
    state = app_state(request)

    async def stream() -> AsyncIterator[str]:
        # 两条唤醒源、一条数据路：已提交帧只从 DB tail 读(notify 或 poll 唤醒);
        # 预览帧读 EphemeralEventBus 订阅。没有进程内直投，也就不需要去重。
        notify_key, notify_queue = state.manager.signal_bus.subscribe_event(run.id)
        preview_subscription = await _subscribe_preview(state, run.id)
        last_seq = _resume_seq(request)
        last_ping = asyncio.get_running_loop().time()

        try:
            while True:
                rows = await state.manager.event_store.after(run.id, last_seq)
                frames, last_seq, saw_done = _drain(rows, last_seq)
                for frame in frames:
                    yield frame
                if saw_done:
                    state.buffer.close(run.id)
                    return
                if not rows:
                    frames, last_seq, settled = await _terminate_if_settled(state, run.id, last_seq)
                    for frame in frames:
                        yield frame
                    if settled:
                        state.buffer.close(run.id)
                        return
                texts, last_ping = await _next_outputs(
                    notify_queue, preview_subscription, last_ping
                )
                for text in texts:
                    yield text
        finally:
            state.manager.signal_bus.unsubscribe("event_committed", notify_key)
            if preview_subscription is not None:
                await preview_subscription.close()

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _terminate_if_settled(
    state: Any, run_id: str, last_seq: int
) -> tuple[list[str], int, bool]:
    """done 帧可能整批丢在失败的 flush 里，行的终态才是权威。

    合成前先重读一次 tail，避免合成帧排在刚写入数据库的 done 之前；仍读不到才合成。
    """
    frames, last_seq, saw_done = _drain(
        await state.manager.event_store.after(run_id, last_seq), last_seq
    )
    if saw_done:
        return frames, last_seq, True
    terminal = await state.manager.terminal_state(run_id)
    if terminal is None:
        return frames, last_seq, False
    # 合成帧的 seq 必须严格大于最后已投帧：前端按已收帧的最大 seq 去重，
    # 同号帧会被当作回放丢弃，合成的 done 帧反而送不到客户端。
    synthesized = project(
        {
            "run_id": run_id,
            "event_type": EventName.RUN_DONE,
            "seq": last_seq + 1,
            "payload": terminal,
        }
    )
    if synthesized is not None:
        frames.append(_sse(synthesized))
    return frames, last_seq, True


def _drain(rows: Any, last_seq: int) -> tuple[list[str], int, bool]:
    """把 DB 行投成 SSE 帧；返回帧、最新 last_seq 与是否见到 done。"""
    frames: list[str] = []
    saw_done = False
    for row in rows:
        last_seq = row.seq
        frame = project(row.record)
        if frame is not None:
            frames.append(_sse(frame))
            saw_done = saw_done or frame.event == "done"
    return frames, last_seq, saw_done


async def _next_outputs(
    notify_queue: Any, preview_subscription: Any, last_ping: float
) -> tuple[list[str], float]:
    """等一轮唤醒，返回本轮要发的 SSE 文本与新的心跳起点。

    没有预览帧时按需产出一条心跳；超时是常态路径，不作为异常。
    """
    items = await _wait_next_frames(notify_queue, preview_subscription)
    if not items:
        now = asyncio.get_running_loop().time()
        line = _heartbeat_line(last_ping, now)
        if line is None:
            return [], last_ping
        return [line], now
    texts: list[str] = []
    for item in items:
        # 预览总线只投 text_delta，由 preview_event 白名单校验保证；
        # project() 默认拒绝，非白名单帧返回 None。
        frame = project(item)
        if frame is not None:
            texts.append(_sse(frame))
    return texts, last_ping


async def _wait_next_frames(
    notify_queue: asyncio.Queue[Any],
    preview_subscription: Any,
) -> list[Any]:
    """等一次唤醒，返回本轮可投的预览帧；超时或只收到 notify 时返回空列表。

    两条唤醒源与一个超时竞争，谁先到用谁。挂起的读取任务必须在本轮结束前回收：
    客户端断连时 asyncio.wait 不会代管子任务，漏回收就是泄漏。
    """
    notify_wait = asyncio.create_task(notify_queue.get())
    waits: list[asyncio.Task[Any]] = [notify_wait]
    preview_wait: asyncio.Task[Any] | None = None
    if preview_subscription is not None:
        preview_wait = asyncio.create_task(preview_subscription.queue.get())
        waits.append(preview_wait)
    done: set[asyncio.Task[Any]] = set()
    try:
        done, _ = await asyncio.wait(
            waits,
            timeout=SSE_DB_POLL_SECONDS,
            return_when=asyncio.FIRST_COMPLETED,
        )
    finally:
        for waiter in waits:
            waiter.cancel()
        await asyncio.gather(*waits, return_exceptions=True)
    items: list[Any] = []
    if preview_wait is not None and preview_wait in done:
        items.append(preview_wait.result())
    if notify_wait in done:
        # notify 只是唤醒信号，payload 是 None，数据由下一轮循环头的 tail 取。
        notify_wait.result()
    return items


def _heartbeat_line(last_ping: float, now: float) -> str | None:
    """到点才给心跳行；未到点返回 None，由调用方继续轮询。"""
    return ": ping\n\n" if now - last_ping >= SSE_HEARTBEAT_SECONDS else None


def _resume_seq(request: Request) -> int:
    """重连带 Last-Event-ID 时从已收序号续放；缺失或非法一律从头回放。"""
    try:
        return max(0, int(request.headers.get("last-event-id", "0")))
    except ValueError:
        return 0


async def _subscribe_preview(state: Any, run_id: str) -> Any | None:
    """订阅预览总线；没有总线或订阅失败都退回纯 DB 流。

    text_delta 是唯一的低延迟通道，但它丢了不影响正确性，主链路不该跟着失败。
    """
    if state.ephemeral_bus is None:
        return None
    try:
        return await state.ephemeral_bus.subscribe(run_id)
    except Exception:  # noqa: BLE001 - durable DB stream remains available
        logger.warning("preview_subscribe_failed run_id=%s", run_id, exc_info=True)
        return None
