"""FastAPI lifespan composition for the HTTP control plane."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from deepresearcher.config import Settings, get_settings
from deepresearcher.service.auth import TokenCodec, make_current_user
from deepresearcher.service.execution.coordinator import WorkerCoordinator
from deepresearcher.service.runs.manager import RunManager
from deepresearcher.service.runtime_stack import build_runtime_stack
from deepresearcher.service.settings import ServiceConfig, get_service_config
from deepresearcher.service.signals import SignalKind
from deepresearcher.service.web.login_rate_limit import LoginRateLimiter


def make_lifespan(
    settings: Settings | None,
    config: ServiceConfig | None,
    graph_factory: Callable[..., Any],
) -> Callable[[FastAPI], Any]:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        service_config = config or get_service_config()
        engine_settings = settings or get_settings()
        # 两进程的真实差异压缩成三个开关:embedded 才需 material/http,
        # 仅分进程模式才需 redis 预览总线(嵌入模式与 Worker 共享进程内 Preview/Hub)。
        async with build_runtime_stack(
            service_config,
            engine_settings,
            with_material=service_config.api_embedded_worker,
            with_http=service_config.api_embedded_worker,
            with_preview_bus=not service_config.api_embedded_worker,
        ) as stack:
            session_factory = stack.session_factory
            signal_bus = stack.signal_bus
            hub = stack.hub
            preview = stack.preview
            material_store = stack.material_store
            http_client = stack.http_client
            checkpointer = stack.checkpointer
            ephemeral_bus = stack.ephemeral_bus
            execution = None
            if service_config.api_embedded_worker:
                execution = WorkerCoordinator(
                    settings=engine_settings,
                    session_factory=session_factory,
                    config=service_config,
                    hub=hub,
                    preview=preview,
                    http_client=http_client,
                    graph_factory=graph_factory,
                    checkpointer=checkpointer,
                    material_store=material_store,
                    ephemeral_bus=ephemeral_bus,
                )
            manager = RunManager(
                session_factory=session_factory,
                config=service_config,
                hub=hub,
                checkpointer=checkpointer,
                signal_bus=signal_bus,
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
                # 与独立 Worker 不同,embedded 的 reconcile 不持启动恢复 advisory
                # lock:部署形态假设两种进程互斥;若将来允许混跑,这里要补锁。
                killed, resumable = await execution.reconcile_startup()
                if killed:
                    app.state.service_logger.info("reconciled_stale_runs count=%d", killed)
                resumed = await execution.resume_runs(resumable)
                if resumed:
                    app.state.service_logger.info("resuming_orphan_runs count=%d", resumed)
                await execution.start()

            app.state.config = service_config
            app.state.settings = engine_settings
            app.state.engine = stack.engine
            app.state.session_factory = session_factory
            app.state.hub = hub
            app.state.preview = preview
            app.state.manager = manager
            app.state.execution = execution
            app.state.checkpointer = checkpointer
            app.state.material_store = material_store
            app.state.ephemeral_bus = ephemeral_bus
            app.state.codec = TokenCodec(service_config.jwt_secret, service_config.token_ttl_hours)
            app.state.login_rate_limiter = LoginRateLimiter(
                session_factory,
                secret=service_config.jwt_secret,
                account_attempts=service_config.login_account_attempts,
                ip_attempts=service_config.login_ip_attempts,
                window_seconds=service_config.login_rate_window_seconds,
                block_seconds=service_config.login_block_seconds,
            )
            app.state.auth_dependency = make_current_user(app.state.codec, session_factory)
            try:
                yield
            finally:
                for kind, key in worker_signal_subscriptions:
                    signal_bus.unsubscribe(kind, key)
                if execution is not None:
                    await execution.shutdown()

    return lifespan
