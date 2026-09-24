"""Independent Worker process lifecycle.

The API owns HTTP/auth/SSE.  This runtime owns graph execution resources and
autonomously consumes the durable PostgreSQL queue.

基础设施装配在 ``service.runtime_stack``(与 API runtime 共享,role="worker" 领取
material/http)。这里保留的 Worker-only 差异:启动恢复持 advisory lock、
coordinator 的订阅与 shutdown。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import text

from deepresearcher.config import Settings, get_settings
from deepresearcher.graph import build_graph
from deepresearcher.observability.logging_config import get_logger
from deepresearcher.service.coordination import WORKER_STARTUP_RECOVERY_LOCK_ID
from deepresearcher.service.execution.coordinator import WorkerCoordinator
from deepresearcher.service.runtime_stack import build_runtime_stack
from deepresearcher.service.settings import ServiceConfig, get_service_config

logger = get_logger("deepresearcher.service.execution.runtime")


@asynccontextmanager
async def _startup_recovery_lock(session_factory: Callable[[], Any]) -> AsyncIterator[None]:
    """跨 Worker 进程串行化启动归类。

    claim 本身已由行锁/CAS 保护。本锁只保护对 ``interrupted`` 行的
    一次性扫描——它可能产生终态事件，因此不得跑两次。
    """
    async with session_factory() as session:
        if session.get_bind().dialect.name != "postgresql":
            yield
            return
        await session.execute(
            text("SELECT pg_advisory_lock(:lock_id)"),
            {"lock_id": WORKER_STARTUP_RECOVERY_LOCK_ID},
        )
        try:
            yield
        finally:
            await session.execute(
                text("SELECT pg_advisory_unlock(:lock_id)"),
                {"lock_id": WORKER_STARTUP_RECOVERY_LOCK_ID},
            )


@asynccontextmanager
async def worker_lifespan(
    settings: Settings | None = None,
    config: ServiceConfig | None = None,
    *,
    graph_factory: Callable[..., Any] = build_graph,
) -> AsyncIterator[WorkerCoordinator]:
    """创建一个独立 Worker 进程拥有的全部资源。

    返回:
        WorkerCoordinator: 已进入启动流程的协调器；上下文退出时关闭。
    """
    service_config = config or get_service_config()
    engine_settings = settings or get_settings()
    async with build_runtime_stack(service_config, engine_settings, role="worker") as stack:
        worker = WorkerCoordinator(
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
