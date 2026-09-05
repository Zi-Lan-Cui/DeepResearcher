"""API control plane for durable Run commands.

The manager accepts, cancels, and resumes runs. Worker-only scheduling,
execution, capacity gates, and crash recovery live in ``WorkerCoordinator``.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, update

from deepsearch_agent.config import Settings
from deepsearch_agent.orchestration.graph import build_graph
from deepsearch_agent.service.event_publisher import RunEventPublisher
from deepsearch_agent.service.events.notifier import EventNotifier
from deepsearch_agent.service.events.store import RunEventStore
from deepsearch_agent.service.events.stream import FanoutSink
from deepsearch_agent.service.models import Run
from deepsearch_agent.service.runs.service import QuotaExceededError as QuotaExceededError
from deepsearch_agent.service.runs.service import RunService
from deepsearch_agent.service.settings import ServiceConfig
from deepsearch_agent.service.worker_service import WorkerCoordinator
from deepsearch_agent.tools.cache import ToolCache

TERMINAL_STATUSES = ("completed", "failed", "cancelled")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class RunManager:
    """Persist user commands and expose durable events to the HTTP layer."""

    def __init__(
        self,
        *,
        settings: Settings | None,
        session_factory: Callable[[], Any],
        config: ServiceConfig,
        fanout: FanoutSink,
        http_client: Any = None,
        graph_factory: Callable[..., Any] = build_graph,
        checkpointer: Any = None,
        event_notifier: EventNotifier | None = None,
        tool_cache: ToolCache | None = None,
        enable_worker: bool = True,
    ) -> None:
        self._checkpointer = checkpointer
        self._session_factory = session_factory
        self._config = config
        self._fanout = fanout
        self.event_notifier = event_notifier or EventNotifier()
        self.event_store = RunEventStore(
            session_factory,
            publish_persisted=fanout.publish_persisted,
            notifier=self.event_notifier,
        )
        self._event_publisher = RunEventPublisher(
            session_factory=session_factory,
            fanout=fanout,
            event_store=self.event_store,
        )
        self._run_service = RunService(session_factory=session_factory, config=config)
        self._execution: WorkerCoordinator | None = None
        if enable_worker:
            if settings is None or http_client is None:
                raise ValueError("Worker execution requires settings and http_client")
            self._execution = WorkerCoordinator(
                settings=settings,
                session_factory=session_factory,
                config=config,
                fanout=fanout,
                event_store=self.event_store,
                event_publisher=self._event_publisher,
                http_client=http_client,
                graph_factory=graph_factory,
                checkpointer=checkpointer,
                tool_cache=tool_cache,
            )

        # Compatibility seams retained until embedded-mode callers migrate.
        self.usage_store = self._execution.usage_store if self._execution else None
        self.llm_gate = self._execution.llm_gate if self._execution else None
        self.llm_rate_limiter = self._execution.llm_rate_limiter if self._execution else None
        self._queue = self._execution.queue if self._execution else None
        self._executor = self._execution.executor if self._execution else None
        self._worker = self._execution.worker if self._execution else None
        self._tasks = self._execution.tasks if self._execution else {}

    @property
    def worker_id(self) -> str | None:
        return self._execution.worker_id if self._execution else None

    async def start_worker(self) -> None:
        if self._execution:
            await self._execution.start()

    async def start(self, user_id: int, query: str) -> str:
        run_id = await self._run_service.create(user_id, query.strip())
        self._fanout.open(run_id)
        await self._event_publisher.publish_status(run_id, "queued")
        await self._event_publisher.flush(run_id)
        if self._execution:
            await self._execution.wake()
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
        elif self._execution:
            self._execution.request_cancel(run_id)
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
        if self._execution:
            await self._execution.wake()
        async with self._session_factory() as session:
            current = await session.get(Run, run_id)
        return current.status if current is not None else "queued"

    # Compatibility proxies retained until callers migrate to WorkerCoordinator.
    async def reconcile_startup(self) -> tuple[int, list[tuple[str, int, str]]]:
        if self._execution is None:
            raise RuntimeError("startup reconciliation requires a Worker runtime")
        return await self._execution.reconcile_startup()

    async def resume_runs(self, pending: list[tuple[str, int, str]]) -> int:
        if self._execution is None:
            raise RuntimeError("run recovery requires a Worker runtime")
        return await self._execution.resume_runs(pending)

    async def shutdown(self) -> None:
        if self._execution:
            await self._execution.shutdown()
