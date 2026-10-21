"""API 与 Worker 进程共用的基础设施装配。

material_store 与 http_client 只有 worker 需要，不在此列。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from deepresearcher.config import Settings
from deepresearcher.service.events.hub import RunEventHub
from deepresearcher.service.events.store import RunEventStore
from deepresearcher.service.persistence.database import (
    make_engine,
    make_session_factory,
    migrate_database,
)
from deepresearcher.service.persistence.redis_material_store import (
    create_research_material_store,
)
from deepresearcher.service.preview.protocol import EphemeralEventBus
from deepresearcher.service.preview.redis import create_redis_ephemeral_bus
from deepresearcher.service.settings import ServiceConfig, checkpoint_dsn
from deepresearcher.service.signals import PostgresSignalBus
from deepresearcher.tools.transport import HttpClient


@dataclass
class RuntimeStack:
    """本进程用到的 engine、session、事件与信号组件。"""

    engine: Any
    session_factory: async_sessionmaker[AsyncSession]
    hub: RunEventHub
    signal_bus: PostgresSignalBus
    checkpointer: Any
    ephemeral_bus: EphemeralEventBus | None
    material_store: Any
    http_client: HttpClient | None


Role = Literal["api", "worker"]


@asynccontextmanager
async def build_runtime_stack(
    service_config: ServiceConfig,
    settings: Settings,
    *,
    role: Role,
) -> AsyncIterator[RuntimeStack]:
    """装配两种 runtime 共享的基础设施，退出时按相反顺序释放。

    每获取一个资源就登记进 AsyncExitStack，装配半途抛错时也按相反顺序释放。
    """
    async with AsyncExitStack() as resources:
        await migrate_database(service_config.database_url)
        engine = make_engine(service_config.database_url)
        resources.push_async_callback(engine.dispose)
        session_factory = make_session_factory(engine)
        signal_bus = PostgresSignalBus()
        await signal_bus.start(service_config.database_url)
        resources.push_async_callback(signal_bus.close)

        material_store = None
        if role == "worker":
            material_store = await create_research_material_store(
                backend=service_config.material_store_backend,
                redis_url=service_config.material_redis_url,
                key_prefix=service_config.material_key_prefix,
                search_ttl_seconds=service_config.search_material_ttl_seconds,
                document_ttl_seconds=service_config.document_material_ttl_seconds,
            )
            resources.push_async_callback(material_store.close)
        http_client = HttpClient(settings.search) if role == "worker" else None
        if http_client is not None:
            resources.push_async_callback(http_client.aclose)
        ephemeral_bus: EphemeralEventBus | None = None
        if service_config.redis_preview_enabled:
            ephemeral_bus = await create_redis_ephemeral_bus(
                service_config.redis_url,
                channel_prefix=service_config.redis_channel_prefix,
                queue_size=service_config.redis_preview_queue_size,
            )
            if ephemeral_bus is not None:  # 工厂可能返回 None；持久事件仍可用，无需登记释放
                resources.push_async_callback(ephemeral_bus.close)

        checkpointer = None
        dsn = checkpoint_dsn(service_config.database_url)
        if dsn is not None:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            from deepresearcher.service.checkpoint_serde import build_checkpointer_serde

            checkpointer = await resources.enter_async_context(
                AsyncPostgresSaver.from_conn_string(dsn, serde=build_checkpointer_serde())
            )
            await checkpointer.setup()

        hub = RunEventHub(
            session_factory=session_factory,
            store=RunEventStore(session_factory),
            signal_bus=signal_bus,
        )
        yield RuntimeStack(
            engine=engine,
            session_factory=session_factory,
            hub=hub,
            signal_bus=signal_bus,
            checkpointer=checkpointer,
            ephemeral_bus=ephemeral_bus,
            material_store=material_store,
            http_client=http_client,
        )
