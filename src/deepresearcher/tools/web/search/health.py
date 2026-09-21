"""搜索提供方的**账户级健康**（鉴权/额度），跨进程共享用。

区别于 `SearchService` 里对 429/503 的**进程内限流窗口**（那是瞬时可自愈、按 key 计数）：
这里表达的是"同一账号 key 已不可用"（401/403/402/432）——重试同 key 无意义，
应让**所有 worker** 在窗口内停止出网撞它，直到人工换 key / 额度恢复。

默认内存实现给单进程与测试；多进程由 `PostgresProviderHealth` 覆盖（跨 worker 共享）。
"""

from __future__ import annotations

import time
from typing import Protocol


class ProviderHealth(Protocol):
    async def is_open(self, provider: str) -> str | None:
        """返回打开原因（invalid_key/quota_exhausted/...），未打开返回 None。"""

    async def trip(self, provider: str, reason: str, seconds: float) -> None:
        """把某 provider 标为不可用一段时间（取更长的窗口）。"""


class MemoryProviderHealth:
    def __init__(self) -> None:
        self._open: dict[str, tuple[float, str]] = {}  # provider -> (恢复点, 原因)

    async def is_open(self, provider: str) -> str | None:
        until, reason = self._open.get(provider, (0.0, ""))
        return reason if time.monotonic() < until else None

    async def trip(self, provider: str, reason: str, seconds: float) -> None:
        current = self._open.get(provider, (0.0, reason))
        self._open[provider] = (max(current[0], time.monotonic() + seconds), reason)
