"""工具语义缓存的窄协议；不依赖具体数据库。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class CacheValue:
    value: dict | list
    content_hash: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    cacheable: bool = True


@dataclass(frozen=True)
class CacheResult:
    value: dict | list
    hit: bool
    content_hash: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


class ToolCache(Protocol):
    async def get_or_compute(
        self,
        namespace: str,
        cache_key: str,
        *,
        ttl_seconds: int,
        schema_version: str,
        compute: Callable[[], Awaitable[CacheValue]],
    ) -> CacheResult: ...

    async def delete_expired(self, *, limit: int = 1_000) -> int: ...


class NoOpToolCache:
    async def get_or_compute(
        self,
        namespace: str,
        cache_key: str,
        *,
        ttl_seconds: int,
        schema_version: str,
        compute: Callable[[], Awaitable[CacheValue]],
    ) -> CacheResult:
        del namespace, cache_key, ttl_seconds, schema_version
        value = await compute()
        return CacheResult(
            value=value.value,
            hit=False,
            content_hash=value.content_hash,
            metrics=value.metrics,
        )

    async def delete_expired(self, *, limit: int = 1_000) -> int:
        del limit
        return 0
