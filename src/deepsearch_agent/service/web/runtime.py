"""FastAPI lifespan composition for the HTTP control plane."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from deepsearch_agent.config import Settings, get_settings
from deepsearch_agent.service.auth import TokenCodec, make_current_user
from deepsearch_agent.service.db import make_engine, make_session_factory, migrate_database
from deepsearch_agent.service.events import FanoutSink
from deepsearch_agent.service.notifier import EventNotifier
from deepsearch_agent.service.runs import RunManager
from deepsearch_agent.service.settings import ServiceConfig, checkpoint_dsn, get_service_config
from deepsearch_agent.service.tool_cache import PostgresToolCache
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

        checkpoint_cm = None
        checkpointer = None
        dsn = checkpoint_dsn(cfg.database_url)
        if dsn is not None:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            checkpoint_cm = AsyncPostgresSaver.from_conn_string(dsn)
            checkpointer = await checkpoint_cm.__aenter__()
            await checkpointer.setup()

        manager = RunManager(
            settings=engine_settings,
            session_factory=session_factory,
            config=cfg,
            fanout=fanout,
            http_client=http_client,
            graph_factory=graph_factory,
            checkpointer=checkpointer,
            event_notifier=event_notifier,
            tool_cache=tool_cache,
            enable_worker=cfg.api_embedded_worker,
        )
        if cfg.api_embedded_worker:
            killed, resumable = await manager.reconcile_startup()
            if killed:
                app.state.service_logger.info("reconciled_stale_runs count=%d", killed)
            resumed = await manager.resume_runs(resumable)
            if resumed:
                app.state.service_logger.info("resuming_orphan_runs count=%d", resumed)

        app.state.config = cfg
        app.state.settings = engine_settings
        app.state.engine = engine
        app.state.session_factory = session_factory
        app.state.fanout = fanout
        app.state.manager = manager
        app.state.checkpointer = checkpointer
        app.state.tool_cache = tool_cache
        app.state.codec = TokenCodec(cfg.jwt_secret, cfg.token_ttl_hours)
        app.state.auth_dependency = make_current_user(app.state.codec, session_factory)
        try:
            yield
        finally:
            await manager.shutdown()
            await event_notifier.close()
            if http_client is not None:
                await http_client.aclose()
            if checkpoint_cm is not None:
                await checkpoint_cm.__aexit__(None, None, None)
            await engine.dispose()

    return lifespan
