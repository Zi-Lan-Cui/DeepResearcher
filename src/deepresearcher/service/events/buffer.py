"""RunEventBuffer：进程内事件缓冲与批量写入。"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from deepresearcher.observability.events.names import EventName
from deepresearcher.observability.logging_config import get_logger
from deepresearcher.service.coordination.signals import PostgresSignalBus
from deepresearcher.service.events.store import RunEventStore

logger = get_logger("deepresearcher.service.events")

#: open 登记数量上限
_OPEN_MAX = 2048

#: done 帧的去重键
_DONE_DEDUP_KEY = "terminal"

_FLUSH_BATCH_MAX = 500


class RunEventBuffer:
    def __init__(
        self,
        *,
        store: RunEventStore,
        signal_bus: PostgresSignalBus,
        open_max: int = _OPEN_MAX,
    ) -> None:
        self.store = store
        self._signal_bus = signal_bus
        self._open_max = max(1, open_max)
        self._lock = threading.Lock()
        # 键序沿用 dict 插入序，即 open 的 FIFO 序；超限时回收最早的登记。
        self._open: dict[str, None] = {}
        self._pending: dict[str, list[dict]] = {}
        self._announced_unrouted = False

    def _evict_oldest_locked(self) -> None:
        oldest = next(iter(self._open))
        del self._open[oldest]
        pending = self._pending.pop(oldest, None)
        if pending:
            logger.warning(
                "run_event_pending_dropped_on_evict run_id=%s count=%d", oldest, len(pending)
            )

    def open(self, run_id: str) -> None:
        with self._lock:
            if run_id not in self._open:
                self._open[run_id] = None
            self._pending.setdefault(run_id, [])
            while len(self._open) > self._open_max:
                self._evict_oldest_locked()

    def is_open(self, run_id: str) -> bool:
        with self._lock:
            return run_id in self._open

    def close(self, run_id: str) -> None:
        """摘除该 run 的 open 登记与缓冲。"""
        with self._lock:
            had_open = self._open.pop(run_id, None) is not None
            pending = self._pending.pop(run_id, None)
        if had_open and pending:
            logger.warning(
                "run_event_pending_dropped_on_close run_id=%s count=%d", run_id, len(pending)
            )

    def _drop_unrouted_locked(self, run_id: Any) -> None:
        if not self._announced_unrouted:
            self._announced_unrouted = True
            logger.warning(
                "run_event_buffer_dropped_unrouted：收到无 run_id 或 run 未 open 的事件"
                "（首个，run_id=%r），此类事件不再生成告警。",
                run_id,
            )

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

    def write_status(self, run_id: str, status: str) -> None:
        self.write(
            {"run_id": run_id, "event_type": EventName.RUN_STATUS, "payload": {"status": status}}
        )

    def write_done(
        self,
        run_id: str,
        *,
        status: str,
        answer_mode: str = "",
        report_available: bool = False,
    ) -> None:
        """写入 done 记录。"""
        self.write(
            {
                "run_id": run_id,
                "event_type": EventName.RUN_DONE,
                "dedup_key": _DONE_DEDUP_KEY,
                "payload": {
                    "status": status,
                    "answer_mode": answer_mode,
                    "report_available": report_available,
                },
            }
        )

    async def flush(self, run_id: str) -> bool:
        """排空该 run 的 pending 记录并写入数据库，单批不超过 _FLUSH_BATCH_MAX。

        返回是否全部落库（无待写记录也算完成）。任一批失败即整批回插队首并返回假，
        剩余批次留待调用方按退避重试；成功后才以库中副本为准发 notify。
        """
        while True:
            with self._lock:
                pending = self._pending.get(run_id)
                if not pending:
                    return True
                records = pending[:_FLUSH_BATCH_MAX]
                del pending[: len(records)]
                if not pending:
                    del self._pending[run_id]
            try:
                await self.store.append(run_id, records)
            except Exception:  # noqa: BLE001 - 写入失败要整批保住，重试节奏交给调用方
                logger.warning("run_event_flush_failed run_id=%s", run_id, exc_info=True)
                self._requeue(run_id, records)
                return False
            # 写入 DB 后以库中副本为准：订阅者经 notify + DB tail 取帧，不经进程内直投。
            try:
                await self._signal_bus.notify_event(run_id)
            except Exception:  # noqa: BLE001 - 帧已落库，订阅方还能靠轮询追平，不该判整批失败
                logger.warning("run_event_notify_failed run_id=%s", run_id, exc_info=True)

    def _requeue(self, run_id: str, records: Sequence[dict]) -> None:
        """持久化失败时把批次放回队首，只在该 run 仍持有 open 登记时回插。
        close 之后的迟到批次按契约丢弃，客户端一侧以 SSE 的行终态为准。"""
        with self._lock:
            if run_id not in self._open:
                return
            self._pending.setdefault(run_id, [])[:0] = records
