"""测试用的进程内消息队列：保留「投递 → 领取 → 确认」语义，去掉网络。

它同时实现两个端口（发布与拉取），让用例用一个对象扮演 Producer 与
SimpleConsumer。与生产实现的差别只有传输：投递顺序、可见性、确认之后不重投
这些行为都保留。
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from deepresearcher.service.bootstrap.layers import MessagingPorts
from deepresearcher.service.messaging.dispatch import new_dispatch
from deepresearcher.service.messaging.receiver import DispatchDelivery


def in_process_ports() -> tuple[MessagingPorts, "InProcessDispatchBus"]:
    """同一进程里让 API 与 Worker 共用一套通道，返回端口与断言用的总线。"""
    bus = InProcessDispatchBus()
    return MessagingPorts(publisher=bus, receiver=bus), bus


class InProcessDispatchBus:
    """内存里的分派总线。"""

    def __init__(self, *, receive_timeout: float = 0.05) -> None:
        self.pending: asyncio.Queue[str] = asyncio.Queue()
        self.published: list[str] = []
        self.acked: list[str] = []
        self.deferred: list[tuple[str, int]] = []
        self._receive_timeout = receive_timeout

    # ---- 发布侧 ----

    async def dispatch(
        self,
        run_id: str,
        local_transaction: Callable[[], Awaitable[None]],
    ) -> None:
        # 与生产同序：本地事务成功才让消息可见，失败则一条都不投。
        await local_transaction()
        await self.redispatch(run_id)

    async def redispatch(self, run_id: str) -> None:
        self.published.append(run_id)
        await self.pending.put(run_id)

    async def close(self) -> None:
        return None

    # ---- 拉取侧 ----

    async def receive(self) -> DispatchDelivery | None:
        try:
            run_id = await asyncio.wait_for(self.pending.get(), timeout=self._receive_timeout)
        except TimeoutError:
            return None
        return DispatchDelivery(
            message=new_dispatch(run_id),
            receipt=run_id,
            delivery_attempt=1,
        )

    async def ack(self, delivery: DispatchDelivery) -> None:
        self.acked.append(delivery.message.run_id)

    async def extend_visibility(self, delivery: DispatchDelivery, seconds: int) -> None:
        self.deferred.append((delivery.message.run_id, seconds))

    # ---- 断言辅助 ----

    def waiting(self) -> int:
        """仍在队列里、尚未被领取的消息数。"""
        return self.pending.qsize()
