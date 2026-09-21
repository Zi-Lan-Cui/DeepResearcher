"""进程无关的基础设施栈：API 与 Worker 两种 runtime 的无差异核。

只收"两边逐字相同"的装配与逆序拆解:migrate/engine/session、FanoutSink、
信号总线、checkpointer(serde 白名单)、事件 store/publisher,以及它们的
teardown 顺序。刻意**不收**的差异——留在各自 runtime 里保持可见:
- API 的 auth/login-limiter/RunManager;Worker 的 coordinator/订阅/恢复锁;
- material_store 与 http_client 在 API 仅 embedded-worker 模式创建,Worker 恒创建;
- redis 预览总线在 API 需 `not api_embedded_worker`,Worker 无条件按开关。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from types import TracebackType
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from deepresearcher.config import Settings
from deepresearcher.service.events.ephemeral import EphemeralEventBus
from deepresearcher.service.events.publisher import RunEventPublisher
from deepresearcher.service.events.redis_ephemeral import create_redis_ephemeral_bus
from deepresearcher.service.events.store import RunEventStore
from deepresearcher.service.events.stream import FanoutSink
from deepresearcher.service.persistence.database import (
    make_engine,
    make_session_factory,
    migrate_database,
)
from deepresearcher.service.persistence.redis_material_store import (
    create_research_material_store,
)
from deepresearcher.service.settings import ServiceConfig, checkpoint_dsn
from deepresearcher.service.signals import PostgresSignalBus
from deepresearcher.tools.transport import HttpClient


@dataclass
class RuntimeStack:
    """基础设施栈的持有物;协调对象(coordinator/manager)由各 runtime 自行装配。"""

    engine: Any
    session_factory: async_sessionmaker[AsyncSession]
    fanout: FanoutSink
    signal_bus: PostgresSignalBus
    event_store: RunEventStore
    event_publisher: RunEventPublisher
    checkpointer: Any
    ephemeral_bus: EphemeralEventBus | None
    material_store: Any
    http_client: HttpClient | None

    async def close(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self.ephemeral_bus is not None:
            await self.ephemeral_bus.close()
        if self.material_store is not None:
            await self.material_store.close()
        if self.http_client is not None:
            await self.http_client.aclose()
        await self.signal_bus.close()
        await self.engine.dispose()


@asynccontextmanager
async def build_runtime_stack(
    cfg: ServiceConfig,
    settings: Settings,
    *,
    with_material: bool,
    with_http: bool,
    with_preview_bus: bool,
) -> AsyncIterator[RuntimeStack]:
    """装配两种 runtime 共享的基础设施,退出时按相反顺序释放。

    三个 with_* 开关就是两进程真实差异的最小表达——想合并差异前先看清这里。
    checkpointer 打开的上下文由栈自持并在 teardown 关闭。
    """
    await migrate_database(cfg.database_url)
    engine = make_engine(cfg.database_url)
    session_factory = make_session_factory(engine)
    fanout = FanoutSink(asyncio.get_running_loop())
    signal_bus = PostgresSignalBus()
    await signal_bus.start(cfg.database_url)

    material_store = None
    if with_material:
        material_store = await create_research_material_store(
            backend=cfg.material_store_backend,
            redis_url=cfg.material_redis_url,
            key_prefix=cfg.material_key_prefix,
            search_ttl_seconds=cfg.search_material_ttl_seconds,
            document_ttl_seconds=cfg.document_material_ttl_seconds,
        )
    http_client = HttpClient(settings.search) if with_http else None
    ephemeral_bus: EphemeralEventBus | None = None
    if with_preview_bus and cfg.redis_preview_enabled:
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
    stack = RuntimeStack(
        engine=engine,
        session_factory=session_factory,
        fanout=fanout,
        signal_bus=signal_bus,
        event_store=event_store,
        event_publisher=event_publisher,
        checkpointer=checkpointer,
        ephemeral_bus=ephemeral_bus,
        material_store=material_store,
        http_client=http_client,
    )
    try:
        yield stack
    finally:
        if checkpoint_cm is not None:
            await checkpoint_cm.__aexit__(None, None, None)
        await stack.close(None, None, None)
