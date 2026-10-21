"""进程内预览总线：EphemeralEventBus 协议的同环实现。

只承载 text_delta 预览帧：无 seq、不写库、队满丢最旧。仅用于 worker 与订阅者
同进程同事件循环的场景；生产装配走 Redis 总线，关闭时预览退化、durable 流不受影响。
"""

from __future__ import annotations

import asyncio

from deepresearcher.service.preview.protocol import (
    EphemeralSubscription,
    preview_event,
)


class LocalPreviewBus:
    """同一事件循环内的预览总线；publish/subscribe/close 符合 EphemeralEventBus 协议。"""

    def __init__(self, *, queue_maxsize: int = 256) -> None:
        self._queue_maxsize = queue_maxsize
        self._subs: dict[str, dict[int, asyncio.Queue]] = {}
        self._next_key = 1
        self._closed = False

    async def publish(self, run_id: str, event: dict) -> None:
        """白名单外的帧不进队列；队列满时丢最旧一帧，预览帧本来就可丢弃。"""
        if self._closed:
            return
        safe = preview_event(event, run_id=run_id)
        if safe is None:
            return
        for queue in list(self._subs.get(run_id, {}).values()):
            try:
                queue.put_nowait(safe)
            except asyncio.QueueFull:
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:  # pragma: no cover - put_nowait 刚腾位不会空
                    pass
                queue.put_nowait(safe)

    async def subscribe(self, run_id: str) -> EphemeralSubscription:
        queue: asyncio.Queue = asyncio.Queue(maxsize=self._queue_maxsize)
        key = self._next_key
        self._next_key += 1
        self._subs.setdefault(run_id, {})[key] = queue

        async def close_subscription() -> None:
            self._subs.get(run_id, {}).pop(key, None)

        return EphemeralSubscription(queue=queue, _close=close_subscription)

    async def close(self) -> None:
        """释放全部订阅并停止接收；幂等，迟到的 publish 静默丢弃。"""
        self._closed = True
        self._subs.clear()
