"""Independent Worker process lifecycle.

The API owns HTTP/auth/SSE.  This runtime owns graph execution resources and
autonomously consumes the durable PostgreSQL queue.

基础设施装配在 ``service.runtime_stack``(与 API runtime 共享)。这里保留的
Worker-only 差异:material/http/preview 恒开、启动恢复持 advisory lock、
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
    """Serialize startup classification across Worker processes.

    Claims themselves are already protected by row locks/CAS.  This lock only
    protects the one-off scan of ``interrupted`` rows, which may emit terminal
    events and therefore must not run twice.
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
    """Create all resources owned by one independent Worker process."""
    cfg = config or get_service_config()
    engine_settings = settings or get_settings()
    async with build_runtime_stack(
        cfg,
        engine_settings,
        with_material=True,
        with_http=True,
        with_preview_bus=True,
    ) as stack:
        worker = WorkerCoordinator(
            settings=engine_settings,
            session_factory=stack.session_factory,
            config=cfg,
            hub=stack.hub,
            preview=stack.preview,
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
