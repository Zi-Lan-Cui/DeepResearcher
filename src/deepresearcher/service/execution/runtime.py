"""Independent Worker process lifecycle.

The API owns HTTP/auth/SSE.  This runtime owns graph execution resources and
autonomously consumes the durable PostgreSQL queue.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import text

from deepresearcher.config import Settings, get_settings
from deepresearcher.observability.logger import get_logger
from deepresearcher.orchestration.graph import build_graph
from deepresearcher.service.coordination import WORKER_STARTUP_RECOVERY_LOCK_ID
from deepresearcher.service.events.ephemeral import EphemeralEventBus
from deepresearcher.service.events.publisher import RunEventPublisher
from deepresearcher.service.events.redis_ephemeral import create_redis_ephemeral_bus
from deepresearcher.service.events.store import RunEventStore
from deepresearcher.service.events.stream import FanoutSink
from deepresearcher.service.execution.coordinator import WorkerCoordinator
from deepresearcher.service.persistence.database import (
    make_engine,
    make_session_factory,
    migrate_database,
)
from deepresearcher.service.persistence.redis_material_store import (
    create_research_material_store,
)
from deepresearcher.service.settings import ServiceConfig, checkpoint_dsn, get_service_config
from deepresearcher.service.signals import PostgresSignalBus
from deepresearcher.tools.transport import HttpClient

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
    import asyncio

    cfg = config or get_service_config()
    engine_settings = settings or get_settings()
    await migrate_database(cfg.database_url)
    engine = make_engine(cfg.database_url)
    session_factory = make_session_factory(engine)
    http_client = HttpClient(engine_settings.search)
    fanout = FanoutSink(asyncio.get_running_loop())
    signal_bus = PostgresSignalBus()
    await signal_bus.start(cfg.database_url)
    ephemeral_bus: EphemeralEventBus | None = None
    if cfg.redis_preview_enabled:
        ephemeral_bus = await create_redis_ephemeral_bus(
            cfg.redis_url,
            channel_prefix=cfg.redis_channel_prefix,
            queue_size=cfg.redis_preview_queue_size,
        )
    material_store = await create_research_material_store(
        backend=cfg.material_store_backend,
        redis_url=cfg.material_redis_url,
        key_prefix=cfg.material_key_prefix,
        search_ttl_seconds=cfg.search_material_ttl_seconds,
        document_ttl_seconds=cfg.document_material_ttl_seconds,
    )

    checkpoint_cm = None
    checkpointer = None
    dsn = checkpoint_dsn(cfg.database_url)
    if dsn is not None:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        from deepresearcher.checkpoint_serde import build_checkpointer_serde

        checkpoint_cm = AsyncPostgresSaver.from_conn_string(dsn, serde=build_checkpointer_serde())
        checkpointer = await checkpoint_cm.__aenter__()
        await checkpointer.setup()

    event_store = RunEventStore(
        session_factory,
        publish_persisted=fanout.publish_persisted,
        signal_bus=signal_bus,
    )
    event_publisher = RunEventPublisher(
        session_factory=session_factory,
        fanout=fanout,
        event_store=event_store,
    )
    worker = WorkerCoordinator(
        settings=engine_settings,
        session_factory=session_factory,
        config=cfg,
        fanout=fanout,
        event_store=event_store,
        event_publisher=event_publisher,
        http_client=http_client,
        graph_factory=graph_factory,
        checkpointer=checkpointer,
        material_store=material_store,
        ephemeral_bus=ephemeral_bus,
    )
    work_subscription = signal_bus.subscribe("run_available", lambda _payload: worker.wake())
    cancel_subscription = signal_bus.subscribe(
        "run_cancel_requested", worker.handle_cancel_notification
    )
    try:
        async with _startup_recovery_lock(session_factory):
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
        signal_bus.unsubscribe("run_available", work_subscription)
        signal_bus.unsubscribe("run_cancel_requested", cancel_subscription)
        await worker.shutdown()
        await signal_bus.close()
        if ephemeral_bus is not None:
            await ephemeral_bus.close()
        await material_store.close()
        await http_client.aclose()
        if checkpoint_cm is not None:
            await checkpoint_cm.__aexit__(None, None, None)
        await engine.dispose()
