"""FastAPI lifespan composition for the HTTP control plane.

The API never executes graphs: ``WorkerCoordinator`` lives exclusively in the
independent worker process (``python -m deepresearcher.worker``). This lifespan
assembles the control-plane collaborators over the shared infrastructure stack.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from deepresearcher.config import Settings, get_settings
from deepresearcher.service.auth import TokenCodec, make_current_user
from deepresearcher.service.runs.manager import RunManager
from deepresearcher.service.runtime_stack import build_runtime_stack
from deepresearcher.service.settings import ServiceConfig, get_service_config
from deepresearcher.service.web.login_rate_limit import LoginRateLimiter


def make_lifespan(
    settings: Settings | None,
    config: ServiceConfig | None,
) -> Callable[[FastAPI], Any]:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        service_config = config or get_service_config()
        engine_settings = settings or get_settings()
        async with build_runtime_stack(service_config, engine_settings, role="api") as stack:
            app.state.config = service_config
            app.state.settings = engine_settings
            app.state.engine = stack.engine
            app.state.session_factory = stack.session_factory
            app.state.hub = stack.hub
            app.state.manager = RunManager(
                session_factory=stack.session_factory,
                config=service_config,
                hub=stack.hub,
                checkpointer=stack.checkpointer,
                signal_bus=stack.signal_bus,
            )
            app.state.checkpointer = stack.checkpointer
            app.state.ephemeral_bus = stack.ephemeral_bus
            app.state.codec = TokenCodec(service_config.jwt_secret, service_config.token_ttl_hours)
            app.state.login_rate_limiter = LoginRateLimiter(
                stack.session_factory,
                secret=service_config.jwt_secret,
                account_attempts=service_config.login_account_attempts,
                ip_attempts=service_config.login_ip_attempts,
                window_seconds=service_config.login_rate_window_seconds,
                block_seconds=service_config.login_block_seconds,
            )
            app.state.auth_dependency = make_current_user(app.state.codec, stack.session_factory)
            yield

    return lifespan
