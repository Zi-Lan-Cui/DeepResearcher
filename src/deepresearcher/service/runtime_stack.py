"""进程无关的基础设施栈：API 与 Worker 两种 runtime 的无差异核。

只收"两边逐字相同"的装配与逆序拆解:migrate/engine/session、事件闸口(Hub/Preview)、
信号总线、checkpointer(serde 白名单)、事件 store/publisher,以及它们的
teardown 顺序。刻意**不收**的差异——留在各自 runtime 里保持可见:
- API 的 auth/login-limiter/RunManager;Worker 的 coordinator/订阅/恢复锁;
- material_store 与 http_client 在 API 仅 embedded-worker 模式创建,Worker 恒创建;
- redis 预览总线在 API 需 `not api_embedded_worker`,Worker 无条件按开关。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from deepresearcher.config import Settings
from deepresearcher.service.events.ephemeral import EphemeralEventBus
from deepresearcher.service.events.hub import RunEventHub
from deepresearcher.service.events.preview import LocalPreviewBus
from deepresearcher.service.events.redis_ephemeral import create_redis_ephemeral_bus
from deepresearcher.service.events.store import RunEventStore
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
    hub: RunEventHub
    preview: LocalPreviewBus
    signal_bus: PostgresSignalBus
    checkpointer: Any
    ephemeral_bus: EphemeralEventBus | None
    material_store: Any
    http_client: HttpClient | None


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

    每获取一个资源就立刻登记进 AsyncExitStack——装配半途抛错(信号总线起了、
    checkpointer 没开成之类)也逆序回卷,不留悬挂连接;RuntimeStack 是纯持有物。
    三个 with_* 开关就是两进程真实差异的最小表达——想合并差异前先看清这里。
    """
    async with AsyncExitStack() as resources:
        await migrate_database(cfg.database_url)
        engine = make_engine(cfg.database_url)
        resources.push_async_callback(engine.dispose)
        session_factory = make_session_factory(engine)
        preview = LocalPreviewBus(asyncio.get_running_loop())
        signal_bus = PostgresSignalBus()
        await signal_bus.start(cfg.database_url)
        resources.push_async_callback(signal_bus.close)

        material_store = None
        if with_material:
            material_store = await create_research_material_store(
                backend=cfg.material_store_backend,
                redis_url=cfg.material_redis_url,
                key_prefix=cfg.material_key_prefix,
                search_ttl_seconds=cfg.search_material_ttl_seconds,
                document_ttl_seconds=cfg.document_material_ttl_seconds,
            )
            resources.push_async_callback(material_store.close)
        http_client = HttpClient(settings.search) if with_http else None
        if http_client is not None:
            resources.push_async_callback(http_client.aclose)
        ephemeral_bus: EphemeralEventBus | None = None
        if with_preview_bus and cfg.redis_preview_enabled:
            ephemeral_bus = await create_redis_ephemeral_bus(
                cfg.redis_url,
                channel_prefix=cfg.redis_channel_prefix,
                queue_size=cfg.redis_preview_queue_size,
            )
            if ephemeral_bus is not None:  # 工厂可降级为 None(持久流仍可用),None 无从回卷
                resources.push_async_callback(ephemeral_bus.close)

        checkpointer = None
        dsn = checkpoint_dsn(cfg.database_url)
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
            preview=preview,
            signal_bus=signal_bus,
        )
        yield RuntimeStack(
            engine=engine,
            session_factory=session_factory,
            hub=hub,
            preview=preview,
            signal_bus=signal_bus,
            checkpointer=checkpointer,
            ephemeral_bus=ephemeral_bus,
            material_store=material_store,
            http_client=http_client,
        )
