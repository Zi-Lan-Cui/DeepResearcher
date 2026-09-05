"""进程内低延迟事件分发：FanoutSink 与 CompositeSink。

引擎的 sink 契约只有一个同步方法 ``write(record)``（JsonlSink 同款 duck-typing）。
FanoutSink 在此基础上承担三件事：

1. **同步收集**：write() 只将无 seq 记录放入 pending；RunEventStore 在数据库
   事务中分配 seq 并提交后，才由 publish_persisted() 唤醒本地订阅者。
2. **溢出策略分离**：活订阅队列有界，满则丢最旧并提示截断；持久化
   pending 不丢，SSE 始终能从数据库 tail 补齐。
3. **线程安全**：在事件循环线程直接入队，否则经 call_soon_threadsafe
   回环投递（asyncio.Queue 非线程安全）。
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from deepsearch_agent.observability.logger import get_logger

logger = get_logger("deepsearch_agent.service.events")

#: 订阅队列收到该哨兵即表示 run 已终结，SSE 生成器应结束。
CLOSE_STREAM: Any = object()

_TRUNCATED_EVENT = "stream_truncated"


class FanoutSink:
    def __init__(self, loop: asyncio.AbstractEventLoop, *, queue_maxsize: int = 256):
        self._loop = loop
        self._loop_thread = threading.get_ident()  # 构造必须发生在事件循环线程
        self._queue_maxsize = queue_maxsize
        self._lock = threading.Lock()
        self._next_key = 1
        self._open: set[str] = set()
        self._pending: dict[str, list[dict]] = {}
        self._subs: dict[str, dict[int, asyncio.Queue]] = {}
        self._dropped: dict[tuple[str, int], int] = {}
        self._announced_unrouted = False

    # ---- 生命周期 ----

    def open(self, run_id: str) -> None:
        with self._lock:
            self._open.add(run_id)
            self._pending.setdefault(run_id, [])
            self._subs.setdefault(run_id, {})

    def close(self, run_id: str) -> None:
        """终止该 run 的分发：迟到事件丢弃，订阅者收到 CLOSE_STREAM 哨兵。

        未取走的 pending 一并丢弃——RunManager 的收尾顺序是 take_pending
        （最后一次 flush）→ close，正常路径不会走到丢数据。
        """
        with self._lock:
            self._open.discard(run_id)
            self._pending.pop(run_id, None)
            subscribers = list(self._subs.pop(run_id, {}).values())
        for queue in subscribers:
            self._put(queue, CLOSE_STREAM)

    def is_open(self, run_id: str) -> bool:
        with self._lock:
            return run_id in self._open

    # ---- 引擎契约 ----

    def write(self, record: BaseModel | Mapping[str, Any] | Any) -> None:
        if isinstance(record, BaseModel):
            data = record.model_dump(exclude_none=True)
        elif isinstance(record, Mapping):
            data = dict(record)
        else:
            return
        run_id = data.get("run_id")
        with self._lock:
            if not isinstance(run_id, str) or run_id not in self._open:
                self._drop_unrouted(run_id)
                return
            self._pending.setdefault(run_id, []).append(data)

    # ---- 订阅 ----

    def subscribe(self, run_id: str) -> tuple[int, asyncio.Queue]:
        """Subscribe to committed local records and ephemeral token previews."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=self._queue_maxsize)
        with self._lock:
            if run_id not in self._open:
                queue.put_nowait(CLOSE_STREAM)
                return -1, queue
            key = self._next_key
            self._next_key += 1
            self._subs[run_id][key] = queue
        return key, queue

    def unsubscribe(self, run_id: str, key: int) -> None:
        with self._lock:
            self._subs.get(run_id, {}).pop(key, None)
            self._dropped.pop((run_id, key), None)

    def seed_seq(self, run_id: str, value: int) -> None:
        """Compatibility no-op: M5 moved sequence ownership to RunEventStore."""

    def publish_persisted(self, run_id: str, records: Sequence[dict]) -> None:
        """Deliver records only after their database transaction committed."""
        with self._lock:
            if run_id not in self._open:
                return
            for record in records:
                if threading.get_ident() == self._loop_thread:
                    self._deliver_locked(run_id, record)
                else:
                    self._loop.call_soon_threadsafe(self._deliver_threadsafe, run_id, record)

    def publish_ephemeral(self, run_id: str, record: dict) -> None:
        """只投递、不记账的旁路通道（token 级预览帧专用）。

        ephemeral 帧不发 seq、不进 pending、永不落库/落文件：它承载的是观感
        （逐字预览），事实由稍后的聚合帧（write 通道，带 seq 可回放）终审。
        与正常帧共用同一订阅队列 → 单连接上的交错顺序天然成立；前端以
        "聚合到达即替换预览" 收敛任何时序。订阅者掉线即丢失，属预期。
        """
        with self._lock:
            if run_id not in self._open:
                return
            self._deliver_locked(run_id, dict(record))

    # ---- 持久化排水 ----

    def take_pending(self, run_id: str) -> list[dict]:
        """取走无 seq 事件，交给 RunEventStore 原子编号并持久化。"""
        with self._lock:
            return self._pending.pop(run_id, [])

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

    def _drop_unrouted(self, run_id: Any) -> None:
        if not self._announced_unrouted:
            self._announced_unrouted = True
            logger.warning(
                "fanout_sink_dropped_unrouted：收到无 run_id 或 run 未 open 的事件"
                "（首个，run_id=%r），此类事件不再生成告警。",
                run_id,
            )


class CompositeSink:
    """把同一条记录扇出到多个 sink；任何一路失败都不拖垮其它路。"""

    def __init__(self, *sinks):
        self._sinks = sinks

    def write(self, record: Any) -> None:
        for sink in self._sinks:
            try:
                sink.write(record)
            except Exception:  # noqa: BLE001 - sink 之间必须互相隔离
                logger.warning(
                    "composite_sink_member_failed sink=%s", type(sink).__name__, exc_info=True
                )
