"""API control plane for durable Run commands.

The manager accepts, cancels, and resumes runs. Worker-only scheduling,
execution, capacity gates, and crash recovery live in ``WorkerCoordinator``.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, update

from deepresearcher.service.events.publisher import RunEventPublisher
from deepresearcher.service.events.store import RunEventStore
from deepresearcher.service.events.stream import FanoutSink
from deepresearcher.service.persistence.models import Run
from deepresearcher.service.runs.service import QuotaExceededError as QuotaExceededError
from deepresearcher.service.runs.service import RunService
from deepresearcher.service.settings import ServiceConfig
from deepresearcher.service.signals import PostgresSignalBus

TERMINAL_STATUSES = ("completed", "failed", "cancelled")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class RunManager:
    """Persist user commands and expose durable events to the HTTP layer."""

    def __init__(
        self,
        *,
        session_factory: Callable[[], Any],
        config: ServiceConfig,
        fanout: FanoutSink,
        checkpointer: Any = None,
        signal_bus: PostgresSignalBus | None = None,
        event_store: RunEventStore | None = None,
        event_publisher: RunEventPublisher | None = None,
    ) -> None:
        self._checkpointer = checkpointer
        self._session_factory = session_factory
        self._config = config
        self._fanout = fanout
        self.signal_bus = signal_bus or PostgresSignalBus()
        self.event_store = event_store or RunEventStore(
            session_factory,
            publish_persisted=fanout.publish_persisted,
            signal_bus=self.signal_bus,
        )
        self._event_publisher = event_publisher or RunEventPublisher(
            session_factory=session_factory, fanout=fanout, event_store=self.event_store
        )
        self._run_service = RunService(session_factory=session_factory, config=config)

    async def start(self, user_id: int, query: str) -> str:
        run_id = await self._run_service.create(user_id, query.strip())
        self._fanout.open(run_id)
        await self._event_publisher.publish_status(run_id, "queued")
        await self._event_publisher.flush(run_id)
        await self.signal_bus.notify_work_available()
        return run_id

    async def cancel(self, user_id: int, run_id: str) -> Run:
        immediate = False
        async with self._session_factory() as session:
            run = await session.scalar(
                select(Run).where(Run.id == run_id, Run.user_id == user_id).with_for_update()
            )
            if run is None or run.user_id != user_id:
                raise LookupError(run_id)
            if run.status in TERMINAL_STATUSES:
                return run
            run.cancellation_requested_at = _utcnow()
            if run.status in ("queued", "awaiting_input", "interrupted"):
                immediate = True
                run.status = "cancelled"
                run.terminal_reason = "user_cancelled"
                run.finished_at = _utcnow()
                run.resume_payload = None
                run.lease_owner = None
                run.lease_expires_at = None
            await session.commit()
        if immediate:
            self._fanout.open(run_id)
            await self._event_publisher.publish_done(run_id)
            await self._event_publisher.flush(run_id)
            self._fanout.close(run_id)
        else:
            await self.signal_bus.notify_cancel(run_id)
        async with self._session_factory() as session:
            return await session.get(Run, run_id)

    async def terminal_state(self, run_id: str) -> dict[str, Any] | None:
        """SSE 兜底终止用:run 已终态则返回 done 帧载荷要素,否则 None。

        done 帧可能在失败批次中丢失(flush 已尽力回插,close 竞态仍可能截尾),
        事件的权威副本是行状态——查到这里即该收尾,不让客户端永挂。
        """
        async with self._session_factory() as session:
            run = await session.get(Run, run_id)
        if run is None or run.status not in TERMINAL_STATUSES:
            return None
        return {
            "status": run.status,
            "answer_mode": run.answer_mode or "",
            "report_available": bool(run.report_markdown),
        }

    async def _has_checkpoint(self, run_id: str) -> bool:
        if self._checkpointer is None:
            return False
        tuple_ = await self._checkpointer.aget_tuple(
            {"configurable": {"thread_id": run_id, "checkpoint_ns": ""}}
        )
        return tuple_ is not None

    async def resume_with_input(self, user_id: int, run_id: str, answer: str) -> str:
        answer = answer.strip()
        if not answer:
            raise ValueError("澄清回答不能为空。")
        async with self._session_factory() as session:
            run = await session.get(Run, run_id)
            if run is None or run.user_id != user_id:
                raise LookupError(run_id)
            if run.status != "awaiting_input":
                raise RuntimeError(run.status)
        if not await self._has_checkpoint(run_id):
            raise RuntimeError("checkpoint_missing")
        async with self._session_factory() as session:
            result = await session.execute(
                update(Run)
                .where(
                    Run.id == run_id,
                    Run.user_id == user_id,
                    Run.status == "awaiting_input",
                    Run.resume_payload.is_(None),
                )
                .values(status="queued", resume_payload={"answer": answer})
            )
            await session.commit()
            if result.rowcount != 1:
                raise RuntimeError("already_resumed")
        self._fanout.open(run_id)
        await self._event_publisher.publish_status(run_id, "queued")
        await self._event_publisher.flush(run_id)
        await self.signal_bus.notify_work_available()
        async with self._session_factory() as session:
            current = await session.get(Run, run_id)
        return current.status if current is not None else "queued"
