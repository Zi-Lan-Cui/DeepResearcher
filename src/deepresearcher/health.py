"""Provider 异常统计和冷却管理。"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Cooldown:
    """冷却策略：连续失败 threshold 次后冷却，时间按被踢次数递增。"""

    threshold: int = 3
    base_seconds: float = 60.0
    max_seconds: float = 300.0

    def delay_for(self, strikes: int) -> float:
        return min(self.base_seconds * max(1, strikes), self.max_seconds)


@dataclass(frozen=True)
class Unavailable:
    """可用性查询的结果"""

    reason: str
    retry_after: float


@runtime_checkable
class HealthLedger(Protocol):
    async def unavailable(self, key: str) -> Unavailable | None: ...

    async def record_failure(self, key: str, reason: str, policy: Cooldown) -> None: ...

    async def record_success(self, key: str) -> None: ...

    async def trip(self, key: str, reason: str, seconds: float) -> None: ...


@dataclass
class _Entry:
    # failures 是连续失败数，成功即清零；strikes 是累计被踢次数，决定冷却时长
    # 递增，只在成功时清零 —— 冷却期内不发请求，所以一直坏会越踢越久。
    failures: int = 0
    strikes: int = 0
    open_until: float = 0.0
    reason: str = ""


class MemoryHealthLedger:
    """进程内实现；多 worker 需换成持久化账本。"""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._entries: dict[str, _Entry] = {}

    async def unavailable(self, key: str) -> Unavailable | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        remaining = entry.open_until - self._clock()
        return Unavailable(entry.reason, remaining) if remaining > 0 else None

    async def record_failure(self, key: str, reason: str, policy: Cooldown) -> None:
        entry = self._entries.setdefault(key, _Entry())
        entry.reason = reason
        entry.failures += 1
        if entry.failures < policy.threshold:
            return
        entry.failures = 0
        entry.strikes += 1
        entry.open_until = self._clock() + policy.delay_for(entry.strikes)

    async def record_success(self, key: str) -> None:
        self._entries.pop(key, None)

    async def trip(self, key: str, reason: str, seconds: float) -> None:
        entry = self._entries.setdefault(key, _Entry())
        entry.reason = reason
        entry.open_until = max(entry.open_until, self._clock() + seconds)
