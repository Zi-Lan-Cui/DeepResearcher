"""发布侧：RocketMQ 事务消息的生产者。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Protocol

from rocketmq.grpc_protocol import (
    TransactionResolution,  # pyright: ignore[reportAttributeAccessIssue]
)
from rocketmq.v5.client.client_configuration import ClientConfiguration, Credentials
from rocketmq.v5.model.message import Message
from rocketmq.v5.producer import Producer
from rocketmq.v5.producer.producer import TransactionChecker
from sqlalchemy import select

from deepresearcher.service.messaging.dispatch import (
    decode_dispatch,
    encode_dispatch,
    new_dispatch,
)
from deepresearcher.service.settings import ServiceConfig
from deepresearcher.service.storage.tables import Run

logger = logging.getLogger("deepresearcher.service.messaging.publisher")

# 回查时查库的等待上限：broker 侧本身有回查间隔，这里不能把 gRPC 线程挂太久。
CHECK_TIMEOUT_SECONDS = 2.0


class DispatchPublisher(Protocol):
    """发布端口：把「这个 run 该被执行一次」与本地事务绑在一起。"""

    async def dispatch(
        self,
        run_id: str,
        local_transaction: Callable[[], Awaitable[None]],
    ) -> None:
        """先发半消息，再执行本地事务，最后回报结果。

        参数：
            run_id: 本次要执行的 run。
            local_transaction: 本地事务体，成功即提交、抛异常即回滚。

        抛出：
            Exception: 本地事务的异常原样上抛；半消息发送失败直接抛，此时数据库没有新行。
        """
        ...

    async def redispatch(self, run_id: str) -> None:
        """补发一条普通分派消息，用于「行已存在但消息丢了」的补偿。

        补偿时本地状态已经确定，没有第二个写操作要与消息协调，因此不发事务消息。
        """
        ...

    async def close(self) -> None: ...


class RunExistsChecker(TransactionChecker):
    """事务回查：按消息体里的 run_id 确认本地事务是否已提交。

    只做主键存在性查询；行已取消或已终态同样算存在，业务裁决留给消费入口。
    查不到库时答未决（``TRANSACTION_RESOLUTION_UNSPECIFIED``），由 broker 稍后重查。
    """

    def __init__(self, session_factory: Callable[[], object], loop: asyncio.AbstractEventLoop):
        self._session_factory = session_factory
        self._loop = loop

    async def _exists(self, run_id: str) -> bool:
        async with self._session_factory() as session:  # type: ignore[attr-defined]
            return await session.scalar(select(Run.id).where(Run.id == run_id)) is not None

    def check(self, message: Message) -> TransactionResolution:
        if message.body is None:
            logger.error("dispatch_check_empty_body message_id=%s", message.message_id)
            return TransactionResolution.ROLLBACK
        try:
            payload = decode_dispatch(message.body)
        except ValueError:
            # 载荷坏了就无法定位 run，重投也不会变好。
            logger.error("dispatch_check_undecodable message_id=%s", message.message_id)
            return TransactionResolution.ROLLBACK
        if self._loop.is_closed():
            # 进程正在关闭，查不了库：回未决让 broker 稍后再查，接管的新实例能读到
            # 真实的行状态；此刻回 ROLLBACK 会把已提交的行变成孤儿。
            return TransactionResolution.TRANSACTION_RESOLUTION_UNSPECIFIED
        try:
            future = asyncio.run_coroutine_threadsafe(self._exists(payload.run_id), self._loop)
            exists = future.result(timeout=CHECK_TIMEOUT_SECONDS)
        except Exception:  # noqa: BLE001 - 回查不能抛
            # 读不到库时也回未决：ROLLBACK 会让已提交的行永久停在 queued，COMMIT 则
            # 可能转正一条本地事务从未提交的消息。官方对进行中的事务也要求保持未决。
            # 一直未决的代价是耗尽半事务消息的最大超时时长（默认 4 小时、不可调）后
            # 被强制回滚，那条行此后没有消息指向它，只能人工按 §7 的 SQL 捞回。
            logger.warning("dispatch_check_db_unavailable run_id=%s", payload.run_id, exc_info=True)
            return TransactionResolution.TRANSACTION_RESOLUTION_UNSPECIFIED
        # 行不存在就回滚。本地事务只有一次配额计数与一条 INSERT，正常远快于首次
        # 回查间隔（默认 60s）；若真出现「回查时还没提交完」的误判，那条消息作废，
        # 行提交后只能靠人工补发，代价是一次误判而不是双写。
        return TransactionResolution.COMMIT if exists else TransactionResolution.ROLLBACK


class RocketMQDispatchPublisher:
    """事务消息生产者的生命周期与调用约定。"""

    def __init__(
        self,
        *,
        config: ServiceConfig,
        session_factory: Callable[[], object],
    ) -> None:
        self._config = config
        self._session_factory = session_factory
        self._producer: Producer | None = None

    async def start(self) -> None:
        """启动 producer；SDK 的启动接口是同步阻塞调用，经线程执行。"""
        client_configuration = ClientConfiguration(
            self._config.rocketmq_endpoints,
            Credentials(self._config.rocketmq_access_key, self._config.rocketmq_access_secret),
            self._config.rocketmq_namespace,
        )
        # 启动检查器：broker 回查时查库确认本地事务是否已提交。
        checker = RunExistsChecker(self._session_factory, asyncio.get_running_loop())
        self._producer = Producer(
            client_configuration,
            topics=[self._config.rocketmq_dispatch_topic, self._config.rocketmq_redispatch_topic],
            checker=checker,
        )
        await asyncio.to_thread(self._producer.startup)

    async def redispatch(self, run_id: str) -> None:
        """暂未启用"""
        producer = self._producer
        if producer is None:
            raise RuntimeError("发布端尚未启动。")
        message = Message()
        # 补发走普通消息 topic：topic 的消息类型在创建时固定，事务 topic 只收事务消息。
        message.topic = self._config.rocketmq_redispatch_topic
        message.body = encode_dispatch(new_dispatch(run_id))
        message.keys = run_id
        await asyncio.to_thread(producer.send, message)

    async def close(self) -> None:
        producer, self._producer = self._producer, None
        if producer is not None:
            await asyncio.to_thread(producer.shutdown)

    async def dispatch(
        self,
        run_id: str,
        local_transaction: Callable[[], Awaitable[None]],
    ) -> None:
        producer = self._producer
        if producer is None:
            raise RuntimeError("发布端尚未启动。")
        message = Message()
        message.topic = self._config.rocketmq_dispatch_topic
        message.body = encode_dispatch(new_dispatch(run_id))
        # keys 只供运维排查，不参与路由；路由由 topic 与消费组决定。
        message.keys = run_id
        transaction = producer.begin_transaction()
        # 半消息先写入 broker：这一步失败时数据库还没有行，直接上抛让调用方重试。
        await asyncio.to_thread(producer.send, message, transaction)
        try:
            # 本地事务提交成功即回报 COMMIT，抛异常即回报 ROLLBACK；回报失败不重试，
            await local_transaction()
        except BaseException:
            # 本地事务自己保证已回滚，这里只是让 broker 丢弃半消息；回报失败不能顶掉
            # 原始异常，否则配额超限会被上报成一次网络错误。
            try:
                await asyncio.to_thread(transaction.rollback)
            except Exception:  # noqa: BLE001 - 回报失败交给 broker 回查
                logger.warning("dispatch_rollback_unreported run_id=%s", run_id, exc_info=True)
            raise
        try:
            # 回报失败不上抛，交由 broker 回查
            await asyncio.to_thread(transaction.commit)
        except Exception:  # noqa: BLE001 - 结果未知交给回查，调用方仍应拿到 run_id
            logger.warning("dispatch_commit_unknown run_id=%s", run_id, exc_info=True)


__all__ = [
    "DispatchPublisher",
    "RocketMQDispatchPublisher",
    "RunExistsChecker",
]
