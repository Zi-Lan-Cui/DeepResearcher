"""本进程事件投递总线:已提交帧的本地直推与逐字预览的旁路通道。

纯"放"的半区——不持有席位(open 归 RunEventHub):谁能收、收到什么时机收,
由 Hub 在 flush/close 时显式指挥。订阅者拿到的队列总是活的;是否值得挂
(席位在不在)由调用方查 `RunEventHub.is_open` 决定。

投递契约:
- deliver 只投已提交(带 seq)的记录,线程安全:非循环线程经 call_soon_threadsafe 回环;
- 队列有界,满则丢最旧并注入 stream_truncated 标记,seq 沿用被丢者保持单调;
- close 发 CLOSE_STREAM 哨兵(终结语义);drop 静默拆线(回收语义,不发哨兵)。
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Sequence
from typing import Any

from deepresearcher.observability.logging_config import get_logger

logger = get_logger("deepresearcher.service.events")

#: 订阅队列收到该哨兵即表示 run 已终结，SSE 生成器应结束。
CLOSE_STREAM: Any = object()

_TRUNCATED_EVENT = "stream_truncated"


class LocalPreviewBus:
    def __init__(self, loop: asyncio.AbstractEventLoop, *, queue_maxsize: int = 256):
        self._loop = loop
        self._loop_thread = threading.get_ident()  # 构造必须发生在事件循环线程
        self._queue_maxsize = queue_maxsize
        self._lock = threading.Lock()
        self._next_key = 1
        self._subs: dict[str, dict[int, asyncio.Queue]] = {}
        self._dropped: dict[tuple[str, int], int] = {}

    # ---- 订阅 ----

    def subscribe(self, run_id: str) -> tuple[int, asyncio.Queue]:
        """Subscribe to committed local records and ephemeral token previews."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=self._queue_maxsize)
        with self._lock:
            key = self._next_key
            self._next_key += 1
            self._subs.setdefault(run_id, {})[key] = queue
        return key, queue

    def unsubscribe(self, run_id: str, key: int) -> None:
        with self._lock:
            self._subs.get(run_id, {}).pop(key, None)
            self._dropped.pop((run_id, key), None)

    # ---- 投递 ----

    def deliver(self, run_id: str, records: Sequence[dict]) -> None:
        """Deliver records only after their database transaction committed.

        调用方(RunEventHub.flush)保证"已提交"与"席位仍在"两前提,这里只管送。
        """
        with self._lock:
            for record in records:
                if threading.get_ident() == self._loop_thread:
                    self._deliver_locked(run_id, record)
                else:
                    self._loop.call_soon_threadsafe(self._deliver_threadsafe, run_id, record)

    def publish_ephemeral(self, run_id: str, record: dict) -> None:
        """只投递、不记账的旁路通道（token 级预览帧专用）。

        ephemeral 帧没有 seq、不进 pending、永不落库/落文件：它承载的是观感
        （逐字预览），事实由稍后的聚合帧（持久通道，带 seq 可回放）终审。
        与正常帧共用同一订阅队列 → 单连接上的交错顺序天然成立；前端以
        "聚合到达即替换预览" 收敛任何时序。订阅者掉线即丢失，属预期。
        """
        with self._lock:
            self._deliver_locked(run_id, dict(record))

    # ---- 生命周期 ----

    def close(self, run_id: str) -> None:
        """终结该 run 的本地投递：订阅者收到 CLOSE_STREAM 哨兵后自行退场。"""
        with self._lock:
            subscribers = list(self._subs.pop(run_id, {}).values())
        for queue in subscribers:
            self._put(queue, CLOSE_STREAM)

    def drop(self, run_id: str) -> None:
        """席位被回收（≠ 终结）：静默拆线，绝不发哨兵——在途订阅者只是
        降级为 DB tail 轮询，run 可能还活着，误发 CLOSE_STREAM 会让前端判死。"""
        with self._lock:
            self._subs.pop(run_id, None)

    # ---- 内部 ----

    def _deliver_threadsafe(self, run_id: str, data: dict) -> None:
        with self._lock:
            self._deliver_locked(run_id, data)

    def _deliver_locked(self, run_id: str, data: dict) -> None:
        for key, queue in list(self._subs.get(run_id, {}).items()):
            try:
                queue.put_nowait(data)
            except asyncio.QueueFull:
                # 丢最旧，原位放截断标记；seq 沿用被丢者的，保持单调。
                try:
                    dropped = queue.get_nowait()
                except asyncio.QueueEmpty:  # pragma: no cover - put_nowait 后不会空
                    dropped = None
                marker = {
                    "event_type": _TRUNCATED_EVENT,
                    "run_id": run_id,
                    # ephemeral 帧没有 seq：丢的是谁就借用谁的编号，兜底 0。
                    "seq": (
                        dropped.get("seq")
                        if isinstance(dropped, dict) and "seq" in dropped
                        else data.get("seq", 0)
                    ),
                    "payload": {},
                }
                try:
                    queue.put_nowait(marker)
                except asyncio.QueueFull:  # pragma: no cover - 刚腾出位
                    pass
                self._dropped[(run_id, key)] = self._dropped.get((run_id, key), 0) + 1

    def _put(self, queue: asyncio.Queue, item: Any) -> None:
        if threading.get_ident() == self._loop_thread:
            self._put_nowait_overflow_safe(queue, item)
        else:  # pragma: no cover - close() 仅在循环线程被调用
            self._loop.call_soon_threadsafe(self._put_nowait_overflow_safe, queue, item)

    @staticmethod
    def _put_nowait_overflow_safe(queue: asyncio.Queue, item: Any) -> None:
        try:
            queue.put_nowait(item)
        except asyncio.QueueFull:  # 满也得让哨兵进：先腾一个位
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            queue.put_nowait(item)
