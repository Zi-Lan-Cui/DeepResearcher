"""可丢弃预览的传输契约。

Ephemeral 事件只改善实时观感：没有序号、从不持久化，不得参与
run 状态判定。协议有两个可互换的实现，装配时选定：
``RedisEphemeralEventBus``（跨进程，生产）与 ``LocalPreviewBus``
（同事件循环，单栈 harness）。
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class EphemeralSubscription:
    """由单个 SSE 请求持有的一份有界预览订阅。"""

    queue: asyncio.Queue[dict]
    _close: Callable[[], Awaitable[None]]
    _closed: bool = field(default=False, init=False)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._close()


@runtime_checkable
class EphemeralEventBus(Protocol):
    """尽力而为的预览总线；实现必须自行吸收传输故障。"""

    async def publish(self, run_id: str, event: dict) -> None: ...

    async def subscribe(self, run_id: str) -> EphemeralSubscription: ...

    async def close(self) -> None: ...


def preview_event(value: object, *, run_id: str) -> dict | None:
    """校验并重建可丢弃通道上唯一允许的事件形状。

    参数:
        value: 待检查的原始事件对象。
        run_id: 事件宣称所属的 run。

    返回:
        dict | None: 白名单通过时返回重建后的安全帧，否则 None。
    """

    if not isinstance(value, dict) or value.get("run_id") != run_id:
        return None
    if value.get("event_type") != "text_delta":
        return None
    payload = value.get("payload")
    if not isinstance(payload, dict):
        return None
    channel = str(payload.get("channel") or "")
    text = str(payload.get("text") or "")[:200]
    if channel not in {"router", "clarify", "supervisor", "writer"} or not text:
        return None
    return {
        "run_id": run_id,
        "event_type": "text_delta",
        "payload": {"channel": channel, "text": text},
    }
