"""FastAPI lifespan composition for the HTTP control plane."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from deepresearcher.config import Settings, get_settings
from deepresearcher.service.auth import TokenCodec, make_current_user
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
from deepresearcher.service.runs.manager import RunManager
from deepresearcher.service.settings import ServiceConfig, checkpoint_dsn, get_service_config
from deepresearcher.service.signals import PostgresSignalBus, SignalKind
from deepresearcher.service.web.login_rate_limit import LoginRateLimiter
from deepresearcher.tools.transport import HttpClient


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
        material_store = None
        http_client = None
        if cfg.api_embedded_worker:
            material_store = await create_research_material_store(
                backend=cfg.material_store_backend,
                redis_url=cfg.material_redis_url,
                key_prefix=cfg.material_key_prefix,
                search_ttl_seconds=cfg.search_material_ttl_seconds,
                document_ttl_seconds=cfg.document_material_ttl_seconds,
            )
            http_client = HttpClient(engine_settings.search)
        fanout = FanoutSink(asyncio.get_running_loop())
        signal_bus = PostgresSignalBus()
        await signal_bus.start(cfg.database_url)
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

            from deepresearcher.service.checkpoint_serde import build_checkpointer_serde

            checkpoint_cm = AsyncPostgresSaver.from_conn_string(
                dsn, serde=build_checkpointer_serde()
            )
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
                material_store=material_store,
            )
        manager = RunManager(
            session_factory=session_factory,
            config=cfg,
            fanout=fanout,
            checkpointer=checkpointer,
            signal_bus=signal_bus,
            event_store=event_store,
            event_publisher=event_publisher,
        )
        worker_signal_subscriptions: list[tuple[SignalKind, int]] = []
        if execution is not None:
            worker_signal_subscriptions = [
                (
                    "run_available",
                    signal_bus.subscribe("run_available", lambda _payload: execution.wake()),
                ),
                (
                    "run_cancel_requested",
                    signal_bus.subscribe(
                        "run_cancel_requested", execution.handle_cancel_notification
                    ),
                ),
            ]
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
        app.state.material_store = material_store
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
            for kind, key in worker_signal_subscriptions:
                signal_bus.unsubscribe(kind, key)
            if execution is not None:
                await execution.shutdown()
            await signal_bus.close()
            if ephemeral_bus is not None:
                await ephemeral_bus.close()
            if material_store is not None:
                await material_store.close()
            if http_client is not None:
                await http_client.aclose()
            if checkpoint_cm is not None:
                await checkpoint_cm.__aexit__(None, None, None)
            await engine.dispose()

    return lifespan
