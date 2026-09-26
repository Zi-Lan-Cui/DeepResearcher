"""进程内预览总线:EphemeralEventBus 协议的同环实现。

只承载 text_delta 预览帧——无 seq、不落库、队满丢最旧。已提交事件不经
进程内直投:SSE 从 DB tail run_events,由信号总线 notify 唤醒(权威副本在库里)。

生产装配不构造本类(redis 预览开启用 Redis 总线,关闭则预览退化、durable 流
不受影响);它存在的唯一理由是单栈 harness——测试里 worker 与订阅者同进程、
同事件循环时,用它把执行器的预览帧直接送进订阅队列。契约(白名单校验、丢帧
可容忍、close 幂等)由协议测试覆盖,与 Redis 实现可互换。
"""

from __future__ import annotations

import asyncio

from deepresearcher.service.preview.protocol import (
    EphemeralSubscription,
    preview_event,
)


class LocalPreviewBus:
    """同一事件循环内的预览总线;publish/subscribe/close 符合 EphemeralEventBus 协议。"""

    def __init__(self, *, queue_maxsize: int = 256) -> None:
        self._queue_maxsize = queue_maxsize
        self._subs: dict[str, dict[int, asyncio.Queue]] = {}
        self._next_key = 1
        self._closed = False

    async def publish(self, run_id: str, event: dict) -> None:
        """白名单外的帧不进队列;队列满丢最旧——预览本来就是可丢弃观感。"""
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
        """释放全部订阅并停止接收;幂等,迟到的 publish 静默丢弃。"""
        self._closed = True
        self._subs.clear()
