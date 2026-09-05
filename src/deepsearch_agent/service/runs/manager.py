"""API control plane for durable Run commands.

The manager accepts, cancels, and resumes runs. Worker-only scheduling,
execution, capacity gates, and crash recovery live in ``WorkerCoordinator``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, update

from deepsearch_agent.service.events.notifier import EventNotifier
from deepsearch_agent.service.events.publisher import RunEventPublisher
from deepsearch_agent.service.events.store import RunEventStore
from deepsearch_agent.service.events.stream import FanoutSink
from deepsearch_agent.service.persistence.models import Run
from deepsearch_agent.service.runs.service import QuotaExceededError as QuotaExceededError
from deepsearch_agent.service.runs.service import RunService
from deepsearch_agent.service.settings import ServiceConfig

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
        event_notifier: EventNotifier | None = None,
        event_store: RunEventStore | None = None,
        event_publisher: RunEventPublisher | None = None,
        wake_worker: Callable[[], Awaitable[None]] | None = None,
        cancel_worker: Callable[[str], None] | None = None,
    ) -> None:
        self._checkpointer = checkpointer
        self._session_factory = session_factory
        self._config = config
        self._fanout = fanout
        self.event_notifier = event_notifier or EventNotifier()
        self.event_store = event_store or RunEventStore(
            session_factory, publish_persisted=fanout.publish_persisted, notifier=self.event_notifier
        )
        self._event_publisher = event_publisher or RunEventPublisher(
            session_factory=session_factory, fanout=fanout, event_store=self.event_store
        )
        self._run_service = RunService(session_factory=session_factory, config=config)
        self._wake_worker = wake_worker
        self._cancel_worker = cancel_worker

    async def start(self, user_id: int, query: str) -> str:
        run_id = await self._run_service.create(user_id, query.strip())
        self._fanout.open(run_id)
        await self._event_publisher.publish_status(run_id, "queued")
        await self._event_publisher.flush(run_id)
        if self._wake_worker is not None:
            await self._wake_worker()
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
        elif self._cancel_worker is not None:
            self._cancel_worker(run_id)
        async with self._session_factory() as session:
            return await session.get(Run, run_id)

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
        if self._wake_worker is not None:
            await self._wake_worker()
        async with self._session_factory() as session:
            current = await session.get(Run, run_id)
        return current.status if current is not None else "queued"
