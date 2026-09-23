"""RunEventHub：事件推送的唯一管理"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from deepresearcher.observability.logging_config import get_logger
from deepresearcher.service.events.store import RunEventStore
from deepresearcher.service.persistence.models import Run
from deepresearcher.service.signals import PostgresSignalBus

logger = get_logger("deepresearcher.service.events")

#: open 登记数量上限(API/Worker 进程各自持有):终结路径有 close,
#: "从未被订阅也从未跑完"的 run 由超额回收兜底,常驻集合不许无界增长。
_OPEN_MAX = 2048
_DONE_DEDUP_MAX = 2048


class RunEventHub:
    def __init__(
        self,
        *,
        session_factory: Callable[[], Any],
        store: RunEventStore,
        signal_bus: PostgresSignalBus,
        open_max: int = _OPEN_MAX,
    ) -> None:
        self._session_factory = session_factory
        self.store = store
        self._signal_bus = signal_bus
        self._open_max = max(1, open_max)
        self._lock = threading.Lock()
        # 键序即 open 的 FIFO 序(dict 保序);超限时回收最早的登记。
        self._open: dict[str, None] = {}
        self._pending: dict[str, list[dict]] = {}
        # done 幂等只需覆盖"同进程相邻两次发布"，插入序即 FIFO 淘汰序。
        self._done_published: dict[str, None] = {}
        self._announced_unrouted = False

    def write(self, record: BaseModel | Mapping[str, Any] | Any) -> None:
        """把无 seq 记录放入该 run 的 pending 缓冲；未 open 的 run 丢弃并告警一次。"""
        if isinstance(record, BaseModel):
            data = record.model_dump(exclude_none=True)
        elif isinstance(record, Mapping):
            data = dict(record)
        else:
            return
        run_id = data.get("run_id")
        with self._lock:
            if not isinstance(run_id, str) or run_id not in self._open:
                self._drop_unrouted_locked(run_id)
                return
            self._pending.setdefault(run_id, []).append(data)

    def open(self, run_id: str) -> None:
        with self._lock:
            if run_id not in self._open:
                self._open[run_id] = None
            self._pending.setdefault(run_id, [])
            while len(self._open) > self._open_max:
                self._evict_oldest_locked()

    def _evict_oldest_locked(self) -> None:
        oldest = next(iter(self._open))
        del self._open[oldest]
        self._pending.pop(oldest, None)

    def close(self, run_id: str) -> None:
        """摘除该 run 的 open 登记与缓冲。

        订阅者经 DB tail + notify 判定终止,close 只影响本进程 pending
        的记账。
        未取走的 pending 一并丢弃——终结顺序(take→最后一次 flush→close)
        是 executor 的职责,正常路径走不到丢数据;走到即记 warning。
        """
        with self._lock:
            had_open = self._open.pop(run_id, None) is not None
            pending = self._pending.pop(run_id, None)
        if had_open and pending:
            logger.warning(
                "run_event_pending_dropped_on_close run_id=%s count=%d", run_id, len(pending)
            )

    def is_open(self, run_id: str) -> bool:
        with self._lock:
            return run_id in self._open

    async def publish_status(self, run_id: str, status: str) -> None:
        self.write({"run_id": run_id, "event_type": "run_status", "payload": {"status": status}})

    async def publish_done(self, run_id: str) -> None:
        with self._lock:
            if run_id in self._done_published:
                return
            self._done_published[run_id] = None
            while len(self._done_published) > _DONE_DEDUP_MAX:
                self._done_published.pop(next(iter(self._done_published)))
        async with self._session_factory() as session:
            run = await session.get(Run, run_id)
        status = run.status if run is not None else "failed"
        self.write(
            {
                "run_id": run_id,
                "event_type": "run_done",
                "payload": {
                    "status": status,
                    "answer_mode": (run.answer_mode if run else None) or "",
                    "report_available": bool(run and run.report_markdown),
                },
            }
        )

    async def flush(self, run_id: str) -> None:
        with self._lock:
            records = self._pending.pop(run_id, [])
        if not records:
            return
        try:
            await self.store.append(run_id, records)
        except Exception:
            logger.warning("run_event_flush_failed run_id=%s", run_id, exc_info=True)
            self._requeue(run_id, records)
            return
        # 写入 DB 后以库中副本为准:订阅者经 notify + DB tail 取帧,不经进程内直投。
        await self._signal_bus.notify_event(run_id)

    def _requeue(self, run_id: str, records: Sequence[dict]) -> None:
        """持久化失败时把批次放回队首——只在该 run 仍持有 open 登记时回插;
        close 之后的迟到批次按既有契约丢弃,客户端由 SSE 的行终态兜底收敛。"""
        with self._lock:
            if run_id not in self._open:
                return
            self._pending.setdefault(run_id, [])[:0] = records

    def _drop_unrouted_locked(self, run_id: Any) -> None:
        if not self._announced_unrouted:
            self._announced_unrouted = True
            logger.warning(
                "run_event_hub_dropped_unrouted：收到无 run_id 或 run 未 open 的事件"
                "（首个，run_id=%r），此类事件不再生成告警。",
                run_id,
            )
