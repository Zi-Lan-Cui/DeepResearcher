"""Agent 工具执行期的轻量并发原语。"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


class ToolExecutionGate:
    """写优先的异步共享/独占门，用于工具并发而非业务持久化。"""

    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._shared_count = 0
        self._exclusive = False
        self._waiting_exclusive = 0

    @asynccontextmanager
    async def shared(self) -> AsyncIterator[None]:
        async with self._condition:
            await self._condition.wait_for(
                lambda: not self._exclusive and self._waiting_exclusive == 0
            )
            self._shared_count += 1
        try:
            yield
        finally:
            async with self._condition:
                self._shared_count -= 1
                if self._shared_count == 0:
                    self._condition.notify_all()

    @asynccontextmanager
    async def exclusive(self) -> AsyncIterator[None]:
        async with self._condition:
            self._waiting_exclusive += 1
            try:
                await self._condition.wait_for(
                    lambda: not self._exclusive and self._shared_count == 0
                )
                self._exclusive = True
            finally:
                self._waiting_exclusive -= 1
                # 等待中的独占调用被取消时，唤醒因写优先而暂停的共享调用。
                self._condition.notify_all()
        try:
            yield
        finally:
            async with self._condition:
                self._exclusive = False
                self._condition.notify_all()
