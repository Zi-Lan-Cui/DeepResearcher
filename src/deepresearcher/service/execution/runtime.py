"""worker 执行面 runtime:进程装配、启动恢复与生命周期。

API 拥有 HTTP/鉴权/SSE;本模块拥有图执行资源,并自主消费持久队列。
基础设施装配在 ``service.runtime_stack``(与 API runtime 共享,role="worker"
领取 material/http)。这里保留的 Worker-only 差异:启动恢复持 advisory lock、
运行时的信号订阅与 shutdown。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import func, select

from deepresearcher.config import Settings, get_settings
from deepresearcher.graph import build_graph
from deepresearcher.observability.logging_config import get_logger
from deepresearcher.service.coordination import WORKER_STARTUP_RECOVERY_LOCK_ID
from deepresearcher.service.events.hub import RunEventHub
from deepresearcher.service.execution.executor import RunExecutor, prune_event_jsonl
from deepresearcher.service.execution.queue import PostgresRunQueue, RunWork
from deepresearcher.service.execution.worker import RunWorker
from deepresearcher.service.persistence.advisory_lock import held_session_lock
from deepresearcher.service.persistence.models import Run, RunEvent
from deepresearcher.service.persistence.models import utcnow as _utcnow
from deepresearcher.service.preview.protocol import EphemeralEventBus
from deepresearcher.service.runs.transitions import apply_transition
from deepresearcher.service.runtime_stack import build_runtime_stack
from deepresearcher.service.settings import ServiceConfig, get_service_config
from deepresearcher.service.usage import CapacityGate, ProviderRateLimiter, UsageStore
from deepresearcher.tools.web.materials import ResearchMaterialStore

logger = get_logger("deepresearcher.service.execution.runtime")


class WorkerRuntime:
    """只归 Worker 进程使用的全部资源与恢复策略。"""

    def __init__(
        self,
        *,
        settings: Settings,
        session_factory: Callable[[], Any],
        config: ServiceConfig,
        hub: RunEventHub,
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
        """跨进程取消的快速路径：先回查队列行再行动。"""
        await self.worker.cancel_if_requested(run_id)

    async def shutdown(self) -> None:
        await self.worker.shutdown()

    async def _settle_cancellations(self) -> None:
        """带取消意图却停在 interrupted 的行:终态化 cancelled 并补 done 帧。

        取消意图的落地不依赖 executor 活着——claim/reap 永远跳过带 flag 的行,
        没有这一步,那些 run 会停在 interrupted 直到用户再次取消。
        """
        for settled in await self.queue.settle_cancellations():
            self._hub.open(settled.run_id)
            await self.executor.publish_done(settled.run_id)
            await self.executor.flush_events(settled.run_id)
            self._hub.close(settled.run_id)

    async def reconcile_startup(self) -> tuple[int, list[tuple[str, int, str]]]:
        """在自主轮询开始前归类 queued/interrupted/孤儿 run。

        返回:
            tuple[int, list[tuple[str, int, str]]]:
                (判死行数, 可续跑的 (run_id, user_id, query) 列表)。
        """
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
                apply_transition(run, "recover_dead", now=_utcnow())
                run.error_message = "进程重启导致运行中断，请重新发起。"
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
            # 被取消 flag 排除、或被他 worker 抢走的行会收到误报的恢复提示。
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


@asynccontextmanager
async def _startup_recovery_lock(session_factory: Callable[[], Any]) -> AsyncIterator[None]:
    """跨 Worker 进程串行化启动归类。

    claim 本身已由行锁/CAS 保护。本锁只保护对 ``interrupted`` 行的
    一次性扫描——它可能产生终态事件，因此不得跑两次。
    """
    async with session_factory() as session:
        async with held_session_lock(session, WORKER_STARTUP_RECOVERY_LOCK_ID):
            yield


@asynccontextmanager
async def worker_lifespan(
    settings: Settings | None = None,
    config: ServiceConfig | None = None,
    *,
    graph_factory: Callable[..., Any] = build_graph,
) -> AsyncIterator[WorkerRuntime]:
    """创建一个独立 Worker 进程拥有的全部资源。

    返回:
        WorkerRuntime: 已进入启动流程的 worker 运行时；上下文退出时关闭。
    """
    service_config = config or get_service_config()
    engine_settings = settings or get_settings()
    async with build_runtime_stack(service_config, engine_settings, role="worker") as stack:
        pruned = prune_event_jsonl(
            service_config.service_log_dir / "events", service_config.jsonl_retention_days
        )
        if pruned:
            logger.info("jsonl_event_logs_pruned count=%d", pruned)
        worker = WorkerRuntime(
            settings=engine_settings,
            session_factory=stack.session_factory,
            config=service_config,
            hub=stack.hub,
            http_client=stack.http_client,
            graph_factory=graph_factory,
            checkpointer=stack.checkpointer,
            material_store=stack.material_store,
            ephemeral_bus=stack.ephemeral_bus,
        )
        work_subscription = stack.signal_bus.subscribe(
            "run_available", lambda _payload: worker.wake()
        )
        cancel_subscription = stack.signal_bus.subscribe(
            "run_cancel_requested", worker.handle_cancel_notification
        )
        try:
            async with _startup_recovery_lock(stack.session_factory):
                killed, resumable = await worker.reconcile_startup()
                resumed = await worker.resume_runs(resumable)
            await worker.start()
            logger.info(
                "worker_started worker_id=%s reconciled=%d resumed=%d",
                worker.worker_id,
                killed,
                resumed,
            )
            yield worker
        finally:
            logger.info("worker_stopping worker_id=%s", worker.worker_id)
            stack.signal_bus.unsubscribe("run_available", work_subscription)
            stack.signal_bus.unsubscribe("run_cancel_requested", cancel_subscription)
            await worker.shutdown()
