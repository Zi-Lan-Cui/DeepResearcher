"""Independent Worker process lifecycle.

The API owns HTTP/auth/SSE.  This runtime owns graph execution resources and
autonomously consumes the durable PostgreSQL queue.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import text

from deepsearch_agent.config import Settings, get_settings
from deepsearch_agent.observability.logger import get_logger
from deepsearch_agent.orchestration.graph import build_graph
from deepsearch_agent.service.coordination import WORKER_STARTUP_RECOVERY_LOCK_ID
from deepsearch_agent.service.event_publisher import RunEventPublisher
from deepsearch_agent.service.events.notifier import EventNotifier
from deepsearch_agent.service.events.store import RunEventStore
from deepsearch_agent.service.events.stream import FanoutSink
from deepsearch_agent.service.persistence.database import (
    make_engine,
    make_session_factory,
    migrate_database,
)
from deepsearch_agent.service.persistence.tool_cache import PostgresToolCache
from deepsearch_agent.service.settings import ServiceConfig, checkpoint_dsn, get_service_config
from deepsearch_agent.service.worker_service import WorkerCoordinator
from deepsearch_agent.tools.cache import NoOpToolCache
from deepsearch_agent.tools.transport import HttpClient

logger = get_logger("deepsearch_agent.service.worker_runtime")


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
    event_notifier = EventNotifier()
    await event_notifier.start(cfg.database_url)
    tool_cache = (
        PostgresToolCache(session_factory)
        if engine_settings.tool_cache.enabled
        else NoOpToolCache()
    )
    await tool_cache.delete_expired()

    checkpoint_cm = None
    checkpointer = None
    dsn = checkpoint_dsn(cfg.database_url)
    if dsn is not None:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        checkpoint_cm = AsyncPostgresSaver.from_conn_string(dsn)
        checkpointer = await checkpoint_cm.__aenter__()
        await checkpointer.setup()

    event_store = RunEventStore(
        session_factory,
        publish_persisted=fanout.publish_persisted,
        notifier=event_notifier,
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
        tool_cache=tool_cache,
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
        await worker.shutdown()
        await event_notifier.close()
        await http_client.aclose()
        if checkpoint_cm is not None:
            await checkpoint_cm.__aexit__(None, None, None)
        await engine.dispose()
