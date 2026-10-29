"""拉取侧：RocketMQ 简单消费者的封装。

取一条、交给消费者，按执行结果确认或延长可见性。消费正确性只看数据库：
delivery_attempt 是观测值，不能用来证明上一个 Worker 已经死了。

长轮询有两种「没消息」的收尾，SDK 都不翻成空批次：到期由 proxy 结束流抛
DEADLINE_EXCEEDED；剩余 deadline 不够跑满一轮时 proxy 直接退回并抛 40018。
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Protocol, cast

import grpc
from rocketmq.v5.client.client_configuration import ClientConfiguration, Credentials
from rocketmq.v5.consumer.simple.simple_consumer import SimpleConsumer
from rocketmq.v5.exception import BadRequestException
from rocketmq.v5.model.filter_expression import FilterExpression
from rocketmq.v5.model.message import Message

from deepresearcher.service.messaging.dispatch import DispatchMessage, decode_dispatch
from deepresearcher.service.settings import ServiceConfig

logger = logging.getLogger("deepresearcher.service.messaging.receiver")

# 拉取循环遇到瞬时错误后的退避：broker 不可用时不空转，也不长时间停止拉取。
_PULL_ERROR_BACKOFF_SECONDS = 0.5
# 空轮询后让出控制权：broker 快速返回空批次时，循环里没有别的挂起点。
_EMPTY_PULL_BACKOFF_SECONDS = 0.01
# proxy 判「剩余 deadline 撑不完一次长轮询」时退回请求用的状态码；消息未被取走，重发即可。
_ILLEGAL_POLLING_TIME = 40018


@dataclass(frozen=True)
class DispatchDelivery:
    """一次消息投递：已取回的分派消息与它的回执句柄。

    receipt 只能用于确认与延长可见性，不参与业务判断。
    """

    message: DispatchMessage
    receipt: Message
    delivery_attempt: int


class DispatchReceiver(Protocol):
    """拉取端口：从队列取投递、确认、延长可见性。"""

    async def receive(self) -> DispatchDelivery | None:
        """取一条投递；没有消息时返回 None。"""
        ...

    async def ack(self, delivery: DispatchDelivery) -> None:
        """确认投递：broker 之后不会重投这一条。"""
        ...

    async def extend_visibility(self, delivery: DispatchDelivery, seconds: int) -> None:
        """延长不可见时长：执行仍在进行，别把同一条消息投给别人。"""
        ...

    async def close(self) -> None: ...


class RocketMQDispatchReceiver:
    """简单消费者：共享一个消费者组，常驻一组并发拉取循环，预取队列写满即暂停拉取。"""

    def __init__(self, *, config: ServiceConfig) -> None:
        self._config = config
        self._consumer: SimpleConsumer | None = None
        self._pulls: list[asyncio.Task[None]] = []
        # 预取上限等于拉取循环数
        self._prefetch: asyncio.Queue[DispatchDelivery] = asyncio.Queue(
            maxsize=config.rocketmq_pull_fanout
        )

    async def start(self) -> None:
        client_configuration = ClientConfiguration(
            self._config.rocketmq_endpoints,
            Credentials(self._config.rocketmq_access_key, self._config.rocketmq_access_secret),
            self._config.rocketmq_namespace,
        )
        consumer = SimpleConsumer(
            client_configuration,
            self._config.rocketmq_consumer_group,
            # 两个 topic 均需订阅：首次受理走事务 topic，补偿补发走普通 topic。
            {
                self._config.rocketmq_dispatch_topic: FilterExpression("*"),
                self._config.rocketmq_redispatch_topic: FilterExpression("*"),
            },
            # 长轮询等待时长：没有消息时挂起，不空转。
            await_duration=self._config.rocketmq_await_duration_seconds,
        )
        self._consumer = consumer
        await asyncio.to_thread(consumer.startup)
        # 创建预拉取任务，每个任务负责从 broker 拉取一条消息并放入预取队列。
        self._pulls = [
            asyncio.create_task(self._pull_forever(), name=f"dispatch-pull-{index}")
            for index in range(self._config.rocketmq_pull_fanout)
        ]

    async def close(self) -> None:
        pulls, self._pulls = self._pulls, []
        for pull in pulls:
            pull.cancel()
        await asyncio.gather(*pulls, return_exceptions=True)
        consumer, self._consumer = self._consumer, None
        if consumer is not None:
            await asyncio.to_thread(consumer.shutdown)

    async def receive(self) -> DispatchDelivery | None:
        """从预取队列取一条；队列空时在这里等（关闭时被取消）。"""
        return await self._prefetch.get()

    async def _pull_forever(self) -> None:
        while True:
            try:
                delivery = await self._pull_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - 瞬时 RPC/解码故障都按「这轮没取到」重试
                logger.warning("dispatch_pull_failed", exc_info=True)
                await asyncio.sleep(_PULL_ERROR_BACKOFF_SECONDS)
                continue
            if delivery is None:
                await asyncio.sleep(_EMPTY_PULL_BACKOFF_SECONDS)
                continue
            await self._prefetch.put(delivery)

    async def _pull_once(self) -> DispatchDelivery | None:
        consumer = self._consumer
        if consumer is None:
            raise RuntimeError("拉取端尚未启动。")

        pending = cast(
            "Future[list[Message]]",
            consumer.receive_async(1, self._config.rocketmq_invisible_seconds),
        )
        try:
            messages = await asyncio.wrap_future(pending)
        except grpc.RpcError as error:
            # 长轮询到期由 proxy 结束流，SDK 不区分「到期」和「其他错误」，只抛 DEADLINE_EXCEEDED。
            if error.code() is grpc.StatusCode.DEADLINE_EXCEEDED:
                return None
            raise
        except BadRequestException as error:
            # 与到期同义
            if error.code == _ILLEGAL_POLLING_TIME:
                return None
            raise
        if not messages:
            return None
        receipt = messages[0]
        if receipt.body is None:
            # 毒消息：无法解析也无法重试，确认后丢弃，避免无限重投占满槽位。
            logger.error("dispatch_received_without_body message_id=%s", receipt.message_id)
            await asyncio.to_thread(consumer.ack, receipt)
            return None
        try:
            message = decode_dispatch(receipt.body)
        except ValueError:
            # 同上：载荷不会因为重投而变好，确认后丢弃。
            logger.error("dispatch_undecodable message_id=%s", receipt.message_id, exc_info=True)
            await asyncio.to_thread(consumer.ack, receipt)
            return None
        return DispatchDelivery(
            message=message,
            receipt=receipt,
            delivery_attempt=int(receipt.delivery_attempt or 0),
        )

    async def ack(self, delivery: DispatchDelivery) -> None:
        consumer = self._consumer
        if consumer is not None:
            await asyncio.to_thread(consumer.ack, delivery.receipt)

    async def extend_visibility(self, delivery: DispatchDelivery, seconds: int) -> None:
        consumer = self._consumer
        if consumer is not None:
            # 用同步版本：SDK 的 *_async 变体没有把 Future 返回出来，拿不到完成信号。
            await asyncio.to_thread(consumer.change_invisible_duration, delivery.receipt, seconds)


__all__ = [
    "DispatchDelivery",
    "DispatchReceiver",
    "RocketMQDispatchReceiver",
]
