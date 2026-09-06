"""FastAPI lifespan composition for the HTTP control plane."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from deepsearch_agent.config import Settings, get_settings
from deepsearch_agent.service.auth import TokenCodec, make_current_user
from deepsearch_agent.service.events.ephemeral import EphemeralEventBus
from deepsearch_agent.service.events.notifier import EventNotifier
from deepsearch_agent.service.events.publisher import RunEventPublisher
from deepsearch_agent.service.events.redis_ephemeral import create_redis_ephemeral_bus
from deepsearch_agent.service.events.store import RunEventStore
from deepsearch_agent.service.events.stream import FanoutSink
from deepsearch_agent.service.execution.coordinator import WorkerCoordinator
from deepsearch_agent.service.persistence.database import (
    make_engine,
    make_session_factory,
    migrate_database,
)
from deepsearch_agent.service.persistence.tool_cache import PostgresToolCache
from deepsearch_agent.service.runs.manager import RunManager
from deepsearch_agent.service.settings import ServiceConfig, checkpoint_dsn, get_service_config
from deepsearch_agent.service.web.login_rate_limit import LoginRateLimiter
from deepsearch_agent.tools.cache import NoOpToolCache
from deepsearch_agent.tools.transport import HttpClient


def make_lifespan(
    settings: Settings | None,
    config: ServiceConfig | None,
    graph_factory: Callable[..., Any],
) -> Callable[[FastAPI], Any]:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        cfg = config or get_service_config()
        engine_settings = settings or get_settings()
        await migrate_database(cfg.database_url)
        engine = make_engine(cfg.database_url)
        session_factory = make_session_factory(engine)
        tool_cache = None
        http_client = None
        if cfg.api_embedded_worker:
            tool_cache = (
                PostgresToolCache(session_factory)
                if engine_settings.tool_cache.enabled
                else NoOpToolCache()
            )
            await tool_cache.delete_expired()
            http_client = HttpClient(engine_settings.search)
        fanout = FanoutSink(asyncio.get_running_loop())
        event_notifier = EventNotifier()
        await event_notifier.start(cfg.database_url)
        ephemeral_bus: EphemeralEventBus | None = None
        # Embedded mode already shares FanoutSink with its Worker. Redis is only
        # needed when API and execution are separate processes.
        if cfg.redis_preview_enabled and not cfg.api_embedded_worker:
            ephemeral_bus = await create_redis_ephemeral_bus(
                cfg.redis_url,
                channel_prefix=cfg.redis_channel_prefix,
                queue_size=cfg.redis_preview_queue_size,
            )

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
        execution = None
        if cfg.api_embedded_worker:
            execution = WorkerCoordinator(
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
        manager = RunManager(
            session_factory=session_factory,
            config=cfg,
            fanout=fanout,
            checkpointer=checkpointer,
            event_notifier=event_notifier,
            event_store=event_store,
            event_publisher=event_publisher,
            wake_worker=execution.wake if execution is not None else None,
            cancel_worker=execution.request_cancel if execution is not None else None,
        )
        if execution is not None:
            killed, resumable = await execution.reconcile_startup()
            if killed:
                app.state.service_logger.info("reconciled_stale_runs count=%d", killed)
            resumed = await execution.resume_runs(resumable)
            if resumed:
                app.state.service_logger.info("resuming_orphan_runs count=%d", resumed)
            await execution.start()

        app.state.config = cfg
        app.state.settings = engine_settings
        app.state.engine = engine
        app.state.session_factory = session_factory
        app.state.fanout = fanout
        app.state.manager = manager
        app.state.execution = execution
        app.state.checkpointer = checkpointer
        app.state.tool_cache = tool_cache
        app.state.ephemeral_bus = ephemeral_bus
        app.state.codec = TokenCodec(cfg.jwt_secret, cfg.token_ttl_hours)
        app.state.login_rate_limiter = LoginRateLimiter(
            session_factory,
            secret=cfg.jwt_secret,
            account_attempts=cfg.login_account_attempts,
            ip_attempts=cfg.login_ip_attempts,
            window_seconds=cfg.login_rate_window_seconds,
            block_seconds=cfg.login_block_seconds,
        )
        app.state.auth_dependency = make_current_user(app.state.codec, session_factory)
        try:
            yield
        finally:
            if execution is not None:
                await execution.shutdown()
            await event_notifier.close()
            if ephemeral_bus is not None:
                await ephemeral_bus.close()
            if http_client is not None:
                await http_client.aclose()
            if checkpoint_cm is not None:
                await checkpoint_cm.__aexit__(None, None, None)
            await engine.dispose()

    return lifespan
