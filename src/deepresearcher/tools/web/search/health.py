"""搜索提供方的账户级健康状态，鉴权与额度类故障，供跨进程共享。

与 SearchService 对 429/503 的进程内限流窗口区分：这里记录 key 已不可用，
对应 401/402/403/432，窗口内所有 worker 停止向该提供方发起请求。
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
        self._open: dict[str, tuple[float, str]] = {}  # provider -> (恢复点， 原因)

    async def is_open(self, provider: str) -> str | None:
        until, reason = self._open.get(provider, (0.0, ""))
        return reason if time.monotonic() < until else None

    async def trip(self, provider: str, reason: str, seconds: float) -> None:
        current = self._open.get(provider, (0.0, reason))
        self._open[provider] = (max(current[0], time.monotonic() + seconds), reason)
