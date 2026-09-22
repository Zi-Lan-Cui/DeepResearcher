"""Worker execution-plane composition, scheduling, and crash recovery."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sqlalchemy import func, select

from deepresearcher.config import Settings
from deepresearcher.graph import build_graph
from deepresearcher.service.events.ephemeral import EphemeralEventBus
from deepresearcher.service.events.hub import RunEventHub
from deepresearcher.service.events.preview import LocalPreviewBus
from deepresearcher.service.execution.executor import RunExecutor
from deepresearcher.service.execution.worker import RunWorker
from deepresearcher.service.persistence.models import Run, RunEvent
from deepresearcher.service.persistence.models import utcnow as _utcnow
from deepresearcher.service.runs.queue import PostgresRunQueue, RunWork
from deepresearcher.service.settings import ServiceConfig
from deepresearcher.service.usage import CapacityGate, ProviderRateLimiter, UsageStore
from deepresearcher.tools.web.materials import ResearchMaterialStore


class WorkerCoordinator:
    """Own all resources and recovery policy used only by a Worker process."""

    def __init__(
        self,
        *,
        settings: Settings,
        session_factory: Callable[[], Any],
        config: ServiceConfig,
        hub: RunEventHub,
        preview: LocalPreviewBus,
        http_client: Any,
        graph_factory: Callable[..., Any] = build_graph,
        checkpointer: Any = None,
        material_store: ResearchMaterialStore | None = None,
        ephemeral_bus: EphemeralEventBus | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._hub = hub
        self._checkpointer = checkpointer
        self.usage_store = UsageStore(session_factory)
        self.llm_gate = CapacityGate(settings.llm.max_concurrent_requests)
        self.llm_rate_limiter = ProviderRateLimiter(
            requests_per_minute=settings.llm.provider_requests_per_minute,
            tokens_per_minute=settings.llm.provider_tokens_per_minute,
        )
        self.queue = PostgresRunQueue(
            session_factory,
            max_global_running=config.max_global_running_runs,
        )
        self.executor = RunExecutor(
            settings=settings,
            session_factory=session_factory,
            config=config,
            hub=hub,
            preview=preview,
            usage_store=self.usage_store,
            llm_gate=self.llm_gate,
            llm_rate_limiter=self.llm_rate_limiter,
            http_client=http_client,
            graph_factory=graph_factory,
            checkpointer=checkpointer,
            material_store=material_store,
            ephemeral_bus=ephemeral_bus,
        )
        self.worker = RunWorker(
            queue=self.queue,
            executor=self.executor,
            max_running=config.max_global_running_runs,
            lease_seconds=config.worker_lease_seconds,
            heartbeat_seconds=config.worker_heartbeat_seconds,
            poll_seconds=config.worker_poll_seconds,
            recover_expired=self._recover_expired,
            after_reap=self._settle_cancellations,
        )

    @property
    def worker_id(self) -> str:
        return self.worker.worker_id

    @property
    def tasks(self) -> dict[str, Any]:
        return self.worker.tasks

    async def start(self) -> None:
        await self.worker.start()

    async def wake(self) -> None:
        await self.worker.wake()

    async def handle_cancel_notification(self, run_id: str) -> None:
        """Fast path for cross-process cancel; the queue row is rechecked first."""
        await self.worker.cancel_if_requested(run_id)

    async def shutdown(self) -> None:
        await self.worker.shutdown()

    async def _settle_cancellations(self) -> None:
        """带取消意图却停在 interrupted 的行:终态化 cancelled 并补 done 帧。

        取消意图的落地不依赖 executor 活着——claim/reap 永远跳过带 flag 的行,
        没有这一步,那些 run 会沉没到用户再点一次取消为止。
        """
        for settled in await self.queue.settle_cancellations():
            self._hub.open(settled.run_id)
            await self.executor.publish_done(settled.run_id)
            await self.executor.flush_events(settled.run_id)
            self._hub.close(settled.run_id)

    async def reconcile_startup(self) -> tuple[int, list[tuple[str, int, str]]]:
        """Classify queued/interrupted/orphaned runs before autonomous polling."""
        resumable: list[tuple[str, int, str]] = []
        killed = 0
        await self.queue.reap_expired()
        await self._settle_cancellations()
        async with self._session_factory() as session:
            stale = (
                await session.scalars(
                    select(Run).where(
                        (Run.status.in_(("queued", "interrupted")))
                        | ((Run.status == "running") & Run.lease_owner.is_(None))
                    )
                )
            ).all()
            for run in stale:
                if run.status == "queued":
                    self._hub.open(run.id)
                    continue
                if await self._has_checkpoint(run.id):
                    resumable.append((run.id, run.user_id, run.query))
                    continue
                killed += 1
                run.status = "failed"
                run.terminal_reason = "server_restart"
                run.error_message = "进程重启导致运行中断，请重新发起。"
                run.finished_at = _utcnow()
                max_seq = await session.scalar(
                    select(func.max(RunEvent.seq)).where(RunEvent.run_id == run.id)
                )
                done_record = {
                    "run_id": run.id,
                    "event_type": "run_done",
                    "seq": int(max_seq or 0) + 1,
                    "payload": {
                        "status": "failed",
                        "answer_mode": run.answer_mode or "",
                        "report_available": False,
                    },
                }
                session.add(
                    RunEvent(
                        run_id=run.id,
                        seq=done_record["seq"],
                        event_type="run_done",
                        record=done_record,
                    )
                )
                run.event_seq = done_record["seq"]
            await session.commit()
        return killed, resumable

    async def _has_checkpoint(self, run_id: str) -> bool:
        if self._checkpointer is None:
            return False
        tuple_ = await self._checkpointer.aget_tuple(
            {"configurable": {"thread_id": run_id, "checkpoint_ns": ""}}
        )
        return tuple_ is not None

    async def _recover_expired(self, work: RunWork) -> RunWork | None:
        self._hub.open(work.run_id)
        if await self._has_checkpoint(work.run_id):
            # "resuming" 帧改由 worker 在 claim 成功后发:此处广播会给
            # 被取消 flag 排除、或被他 worker 抢走的行留下幻影恢复提示。
            return RunWork(
                run_id=work.run_id,
                user_id=work.user_id,
                query=work.query,
                resume=True,
            )
        await self.executor.persist_status(
            work.run_id,
            status="failed",
            terminal_reason="lease_expired_without_checkpoint",
            error_message="运行中断且没有可恢复断点，请重新发起。",
        )
        await self.executor.publish_done(work.run_id)
        await self.executor.flush_events(work.run_id)
        self._hub.close(work.run_id)
        return None

    async def resume_runs(self, pending: list[tuple[str, int, str]]) -> int:
        for run_id, user_id, query in pending:
            # 同 _recover_expired:恢复广播推迟到 claim 成功之后(worker 侧发)。
            self._hub.open(run_id)
            await self.worker.submit(
                RunWork(run_id=run_id, user_id=user_id, query=query, resume=True)
            )
        await self.worker.wake()
        return len(pending)
