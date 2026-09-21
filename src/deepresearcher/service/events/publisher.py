"""Run event publication shared by the API control plane and Worker executor."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from deepresearcher.observability.logger import get_logger
from deepresearcher.service.events.store import RunEventStore
from deepresearcher.service.events.stream import FanoutSink
from deepresearcher.service.persistence.models import Run

logger = get_logger("deepresearcher.service.events.publisher")
_DONE_DEDUP_MAX = 2048


class RunEventPublisher:
    """Buffer service events locally, then persist and wake DB-tail consumers."""

    def __init__(
        self,
        *,
        session_factory: Callable[[], Any],
        fanout: FanoutSink,
        event_store: RunEventStore,
    ) -> None:
        self._session_factory = session_factory
        self._fanout = fanout
        self._event_store = event_store
        # done 幂等只需覆盖"同进程相邻两次发布"，dict 插入序即 FIFO 淘汰序，
        # 常驻集合不允许随 run 数单调增长。
        self._done_published: dict[str, None] = {}

    async def flush(self, run_id: str) -> None:
        pending = self._fanout.take_pending(run_id)
        if not pending:
            return
        try:
            await self._event_store.append(run_id, pending)
        except Exception:  # noqa: BLE001 - event drain must not poison the run
            logger.warning("run_event_flush_failed run_id=%s", run_id, exc_info=True)
            self._fanout.requeue_pending(run_id, pending)

    async def publish_status(self, run_id: str, status: str) -> None:
        self._fanout.write(
            {"run_id": run_id, "event_type": "run_status", "payload": {"status": status}}
        )

    async def publish_done(self, run_id: str) -> None:
        if run_id in self._done_published:
            return
        self._done_published[run_id] = None
        if len(self._done_published) > _DONE_DEDUP_MAX:
            self._done_published.pop(next(iter(self._done_published)))
        async with self._session_factory() as session:
            run = await session.get(Run, run_id)
        status = run.status if run is not None else "failed"
        self._fanout.write(
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
