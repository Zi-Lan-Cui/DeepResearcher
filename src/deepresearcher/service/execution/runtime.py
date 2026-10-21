"""worker 执行层 runtime：进程装配、启动恢复与生命周期。

API 拥有 HTTP、鉴权与 SSE；本模块拥有图执行资源，并自主消费持久队列。
基础设施装配在 service.runtime_stack，与 API runtime 共享，material_store 与
http_client 仅 role="worker" 时创建。本文件保留的 Worker 独有差异：启动恢复
持 advisory lock、运行时的信号订阅与 shutdown。
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import select

from deepresearcher.config import Settings, get_settings
from deepresearcher.graph import build_graph
from deepresearcher.observability.logging_config import get_logger
from deepresearcher.service.coordination import WORKER_STARTUP_RECOVERY_LOCK_ID
from deepresearcher.service.events.hub import RunEventHub
from deepresearcher.service.execution.executor import RunExecutor
from deepresearcher.service.execution.queue import PostgresRunQueue, RunWork
from deepresearcher.service.execution.telemetry import (
    configure_tracer_provider,
    shutdown_tracer_provider,
)
from deepresearcher.service.execution.usage import CapacityGate, ProviderRateLimiter, UsageStore
from deepresearcher.service.execution.worker import RunWorker
from deepresearcher.service.persistence.advisory_lock import held_session_lock
from deepresearcher.service.persistence.models import Run
from deepresearcher.service.persistence.models import utcnow as _utcnow
from deepresearcher.service.preview.protocol import EphemeralEventBus
from deepresearcher.service.retention import prune_event_jsonl, sweep_run_traces
from deepresearcher.service.runs.transitions import apply_transition
from deepresearcher.service.runtime_stack import build_runtime_stack
from deepresearcher.service.settings import ServiceConfig, get_service_config
from deepresearcher.tools.web.materials import ResearchMaterialStore

logger = get_logger("deepresearcher.service.execution.runtime")

# 保留清扫循环首轮延迟：等连接池与启动归类集中处理，随机抖动另防副本齐动。
_FIRST_SWEEP_DELAY_SECONDS = 60


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
        self._config = config
        self._retention_task: asyncio.Task[None] | None = None
        self.usage_store = UsageStore(session_factory)
        self.llm_gate = CapacityGate(settings.llm.max_concurrent_requests)
        self.llm_rate_limiter = ProviderRateLimiter(
            requests_per_minute=settings.llm.provider_requests_per_minute,
            tokens_per_minute=settings.llm.provider_tokens_per_minute,
        )
        self.queue = PostgresRunQueue(session_factory)
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
            max_running=config.worker_max_running_runs,
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
        self._start_retention_loop()

    async def wake(self) -> None:
        await self.worker.wake()

    async def handle_cancel_notification(self, run_id: str) -> None:
        """跨进程取消的快速路径：先回查队列行再行动。"""
        await self.worker.cancel_if_requested(run_id)

    def _start_retention_loop(self) -> None:
        config = self._config
        if config.run_trace_retention_days <= 0 and config.jsonl_retention_days <= 0:
            return
        self._retention_task = asyncio.create_task(
            self._retention_loop(), name=f"retention-sweep-{self.worker.worker_id}"
        )

    async def _retention_loop(self) -> None:
        """保留期清扫的常驻循环：首轮短延迟后错峰，此后按配置间隔。

        首轮不等满一个间隔，让重启也起到清淤作用；随机抖动防多副本齐动。
        """
        config = self._config
        logger.info(
            "retention_loop_started trace_days=%d jsonl_days=%d interval_seconds=%d",
            config.run_trace_retention_days,
            config.jsonl_retention_days,
            config.retention_sweep_interval_seconds,
        )
        delay_seconds = _FIRST_SWEEP_DELAY_SECONDS + random.uniform(0.0, 30.0)
        try:
            while True:
                await asyncio.sleep(delay_seconds)
                delay_seconds = config.retention_sweep_interval_seconds
                try:
                    summary = await sweep_run_traces(
                        self._session_factory,
                        self._checkpointer,
                        retention_days=config.run_trace_retention_days,
                    )
                    pruned_files = prune_event_jsonl(
                        config.service_log_dir / "events", config.jsonl_retention_days
                    )
                except Exception:  # noqa: BLE001 - 瞬断下轮重试，清扫不得波及在线与关停
                    logger.warning("retention_sweep_failed", exc_info=True)
                    continue
                if summary or pruned_files:
                    logger.info("retention_sweep summary=%s jsonl_files=%d", summary, pruned_files)
        except asyncio.CancelledError:
            return

    async def shutdown(self) -> None:
        if self._retention_task is not None:
            self._retention_task.cancel()
            await asyncio.gather(self._retention_task, return_exceptions=True)
            self._retention_task = None
        await self.worker.shutdown()
        self.executor.shutdown_langfuse()

    async def _settle_cancellations(self) -> None:
        """带取消意图却停在 interrupted 的行：终态化 cancelled 并补 done 帧。

        取消意图的执行不依赖 executor 存活：claim/reap 始终跳过带取消标记的行，
        没有这一步，那些 run 会停在 interrupted 直到用户再次取消。
        """
        for settled in await self.queue.settle_cancellations():
            self._hub.open(settled.run_id)
            await self.executor.publish_done(settled.run_id)
            await self.executor.flush_events(settled.run_id)
            self._hub.close(settled.run_id)

    async def reconcile_startup(self) -> tuple[int, list[tuple[str, int, str]]]:
        """在自主轮询开始前归类 queued/interrupted/孤儿 run。

        返回：
            判死的行数，以及可续跑的 (run_id, user_id, query) 列表。
        """
        resumable: list[tuple[str, int, str]] = []
        dead_run_ids: list[str] = []
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
                dead_run_ids.append(run.id)
            await session.commit()
        # done 帧走 hub → store → NOTIFY 这条唯一投递路径，open/publish/flush/close
        # 次序与 _settle_cancellations 相同；publish_done 在事务提交后读行，状态已是终态 failed。
        for run_id in dead_run_ids:
            self._hub.open(run_id)
            await self.executor.publish_done(run_id)
            await self.executor.flush_events(run_id)
            self._hub.close(run_id)
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
            # 不在此处广播 "resuming" 帧，由 worker claim 成功后发：此处广播会让
            # 被取消标记排除、或被他 worker 抢走的行收到误报的恢复提示。
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
            # 同 _recover_expired：恢复广播推迟到 claim 成功之后(worker 侧发)。
            self._hub.open(run_id)
            await self.worker.submit(
                RunWork(run_id=run_id, user_id=user_id, query=query, resume=True)
            )
        await self.worker.wake()
        return len(pending)


@asynccontextmanager
async def _startup_recovery_lock(session_factory: Callable[[], Any]) -> AsyncIterator[None]:
    """跨 Worker 进程串行化启动归类。

    claim 本身已由行锁/CAS 保护。本锁只保护对 interrupted 行的
    一次性扫描——它可能产生终态事件，因此不得执行两次。
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

    返回：
        已进入启动流程的 worker 运行时，上下文退出时释放。
    """
    service_config = config or get_service_config()
    engine_settings = settings or get_settings()
    # 先于 RunExecutor(首次 Langfuse())装配全局 provider，让 langfuse 接管同一棵树。
    configure_tracer_provider(service_config)
    async with build_runtime_stack(service_config, engine_settings, role="worker") as stack:
        # 保留期清扫(jsonl + run 过程数据)统一由 WorkerRuntime 的背景循环承担，
        # 启动路径不再各自做一次性清理。
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
                "worker_started worker_id=%s killed=%d resumed=%d",
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
            shutdown_tracer_provider()
