"""API 与 Worker 进程共用的基础设施装配。

material_store 与 http_client 只有 worker 需要，不在此列。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal

from psycopg import AsyncConnection
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from deepresearcher.config import Settings
from deepresearcher.service.coordination.keys import CHECKPOINT_MIGRATION_LOCK_ID
from deepresearcher.service.coordination.locks import held_autocommit_lock
from deepresearcher.service.coordination.signals import PostgresSignalBus
from deepresearcher.service.events.buffer import RunEventBuffer
from deepresearcher.service.events.store import RunEventStore
from deepresearcher.service.messaging.publisher import DispatchPublisher, RocketMQDispatchPublisher
from deepresearcher.service.messaging.receiver import DispatchReceiver, RocketMQDispatchReceiver
from deepresearcher.service.preview.protocol import EphemeralEventBus
from deepresearcher.service.preview.redis import create_redis_ephemeral_bus
from deepresearcher.service.settings import ServiceConfig, checkpoint_dsn
from deepresearcher.service.storage.engine import (
    make_engine,
    make_session_factory,
    migrate_database,
)
from deepresearcher.tools.transport import HttpClient
from deepresearcher.tools.web.materials.store import (
    create_research_material_store,
)


@dataclass
class MessagingPorts:
    """消息队列的两个端口。

    API 只发布（receiver 为 None），Worker 同时发布与拉取：共用同一个消费者组时，
    API 若也建消费者会与 Worker 抢同一条消息，所以按角色显式区分。
    """

    publisher: DispatchPublisher
    receiver: DispatchReceiver | None


@dataclass
class RuntimeStack:
    """本进程用到的 engine、session、事件与信号组件。"""

    engine: Any
    session_factory: async_sessionmaker[AsyncSession]
    buffer: RunEventBuffer
    signal_bus: PostgresSignalBus
    publisher: DispatchPublisher
    receiver: DispatchReceiver | None
    checkpointer: Any
    ephemeral_bus: EphemeralEventBus | None
    material_store: Any
    http_client: HttpClient | None


Role = Literal["api", "worker"]


async def _start_messaging(
    service_config: ServiceConfig,
    session_factory: async_sessionmaker[AsyncSession],
    *,
    role: Role,
) -> tuple[MessagingPorts, list[Any]]:
    """按角色建立真实的 RocketMQ 端口，返回端口与需要在退出时释放的对象。"""
    publisher = RocketMQDispatchPublisher(config=service_config, session_factory=session_factory)
    await publisher.start()
    closables: list[Any] = [publisher]
    receiver: DispatchReceiver | None = None
    if role == "worker":
        rocketmq_receiver = RocketMQDispatchReceiver(config=service_config)
        try:
            await rocketmq_receiver.start()
        except BaseException:
            await publisher.close()
            raise
        receiver = rocketmq_receiver
        closables.append(rocketmq_receiver)
    return MessagingPorts(publisher=publisher, receiver=receiver), closables


async def _setup_checkpointer(checkpointer: Any, dsn: str) -> None:
    """在 checkpoint 迁移锁内创建 checkpoint 表。"""
    # 独用一把锁，不与 DATABASE_MIGRATION_LOCK_ID 共用：setup 里的 CREATE INDEX CONCURRENTLY
    # 要等所有更早快照的事务结束，而 migrate_database 等同一把锁时会挂着这样一个事务，
    # 共用就成环。连接也必须 autocommit，见 held_autocommit_lock。
    async with await AsyncConnection.connect(dsn, autocommit=True) as connection:
        async with held_autocommit_lock(connection, CHECKPOINT_MIGRATION_LOCK_ID):
            await checkpointer.setup()


@asynccontextmanager
async def build_runtime_stack(
    service_config: ServiceConfig,
    settings: Settings,
    *,
    role: Role,
    messaging: MessagingPorts | None = None,
) -> AsyncIterator[RuntimeStack]:
    """装配两种 runtime 共享的基础设施，退出时按相反顺序释放。

    每获取一个资源就登记进 AsyncExitStack，装配半途抛错时也按相反顺序释放。

    参数：
        messaging: 消息队列端口；为 None 时按配置连真实 RocketMQ。测试注入进程内
            替身时必须成对传入，否则同一个 run 会被两套通道各投一次。
    """
    async with AsyncExitStack() as resources:
        await migrate_database(service_config.database_url)
        engine = make_engine(service_config.database_url)
        resources.push_async_callback(engine.dispose)
        session_factory = make_session_factory(engine)
        signal_bus = PostgresSignalBus()
        await signal_bus.start(service_config.database_url)
        resources.push_async_callback(signal_bus.close)

        # 发布侧：API 用它在受理时发半消息，Worker 用它在崩溃接管时补发。
        ports = messaging
        if ports is None:
            ports, messaging_closables = await _start_messaging(
                service_config, session_factory, role=role
            )
            for closable in reversed(messaging_closables):
                resources.push_async_callback(closable.close)

        material_store = None
        if role == "worker":
            material_store = await create_research_material_store(
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

        dsn = checkpoint_dsn(service_config.database_url)
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        from deepresearcher.service.execution.checkpoint_serde import build_checkpointer_serde

        checkpointer = await resources.enter_async_context(
            AsyncPostgresSaver.from_conn_string(dsn, serde=build_checkpointer_serde())
        )
        await _setup_checkpointer(checkpointer, dsn)

        buffer = RunEventBuffer(
            store=RunEventStore(session_factory),
            signal_bus=signal_bus,
        )
        yield RuntimeStack(
            engine=engine,
            session_factory=session_factory,
            buffer=buffer,
            signal_bus=signal_bus,
            publisher=ports.publisher,
            receiver=ports.receiver,
            checkpointer=checkpointer,
            ephemeral_bus=ephemeral_bus,
            material_store=material_store,
            http_client=http_client,
        )
