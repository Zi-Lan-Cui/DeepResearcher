"""搜索提供方的账户级健康状态，鉴权与额度类故障，供跨进程共享。

与 SearchService 里对 429/503 的进程内限流窗口区分开：限流窗口是瞬时可自愈的，
这里记录的是同一账号 key 已不可用，对应 401/402/403/432。重试同一 key 无意义，
窗口内所有 worker 都应停止向该提供方出网请求，直到人工换 key 或额度恢复。

默认内存实现适用于单进程与测试；多进程部署由 PostgresProviderHealth 跨 worker 共享。
"""

from __future__ import annotations

import time
from typing import Protocol


class ProviderHealth(Protocol):
    async def is_open(self, provider: str) -> str | None:
        """熔断中返回原因字符串，例如 invalid_key 或 quota_exhausted；未熔断返回 None。"""

    async def trip(self, provider: str, reason: str, seconds: float) -> None:
        """把某 provider 标为不可用一段时间；已有更长窗口时不缩短它。"""


class MemoryProviderHealth:
    def __init__(self) -> None:
        self._open: dict[str, tuple[float, str]] = {}  # provider -> (恢复点, 原因)

    async def is_open(self, provider: str) -> str | None:
        until, reason = self._open.get(provider, (0.0, ""))
        return reason if time.monotonic() < until else None

    async def trip(self, provider: str, reason: str, seconds: float) -> None:
        current = self._open.get(provider, (0.0, reason))
        self._open[provider] = (max(current[0], time.monotonic() + seconds), reason)
