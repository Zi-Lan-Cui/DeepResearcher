"""PostgreSQL/SQLite 通用的持久工具缓存。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import delete, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from deepresearcher.observability.usage_runtime import record_cache_event
from deepresearcher.service.persistence.models import ToolCacheEntry
from deepresearcher.tools.cache import CacheResult, CacheValue

logger = logging.getLogger("deepresearcher.service.persistence.tool_cache")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class _Flight:
    lock: asyncio.Lock
    users: int = 0


class PostgresToolCache:
    """数据库持久 + 单进程同键 single-flight。"""

    def __init__(self, session_factory: Callable[[], Any]) -> None:
        self._session_factory = session_factory
        self._flights: dict[tuple[str, str], _Flight] = {}
        self._flights_lock = asyncio.Lock()

    async def get_or_compute(
        self,
        namespace: str,
        cache_key: str,
        *,
        ttl_seconds: int,
        schema_version: str,
        compute: Callable[[], Awaitable[CacheValue]],
    ) -> CacheResult:
        if ttl_seconds <= 0:
            await record_cache_event(namespace=namespace, status="bypass")
            return self._result(await compute(), hit=False)
        cached = await self._get(namespace, cache_key, schema_version)
        if cached is not None:
            await record_cache_event(namespace=namespace, status="hit", detail=cached.metrics)
            return cached

        flight_key = (namespace, cache_key)
        flight = await self._join_flight(flight_key)
        try:
            async with flight.lock:
                cached = await self._get(namespace, cache_key, schema_version)
                if cached is not None:
                    await record_cache_event(
                        namespace=namespace, status="hit", detail=cached.metrics
                    )
                    return cached
                await record_cache_event(namespace=namespace, status="miss")
                value = await compute()
                if not value.cacheable:
                    await record_cache_event(namespace=namespace, status="bypass")
                    return self._result(value, hit=False)
                written = await self._put(
                    namespace,
                    cache_key,
                    value,
                    ttl_seconds=ttl_seconds,
                    schema_version=schema_version,
                )
                await record_cache_event(
                    namespace=namespace, status="write" if written else "bypass"
                )
                return self._result(value, hit=False)
        finally:
            await self._leave_flight(flight_key, flight)

    async def delete_expired(self, *, limit: int = 1_000) -> int:
        if limit <= 0:
            return 0
        try:
            async with self._session_factory() as session:
                keys = (
                    await session.execute(
                        ToolCacheEntry.__table__.select()
                        .with_only_columns(ToolCacheEntry.namespace, ToolCacheEntry.cache_key)
                        .where(ToolCacheEntry.expires_at <= _utcnow())
                        .limit(limit)
                    )
                ).all()
                if not keys:
                    return 0
                for namespace, cache_key in keys:
                    await session.execute(
                        delete(ToolCacheEntry).where(
                            ToolCacheEntry.namespace == namespace,
                            ToolCacheEntry.cache_key == cache_key,
                        )
                    )
                await session.commit()
                return len(keys)
        except Exception:  # noqa: BLE001 - 缓存清理失败不阻断业务
            logger.warning("tool_cache_cleanup_failed", exc_info=True)
            return 0

    async def _get(self, namespace: str, cache_key: str, schema_version: str) -> CacheResult | None:
        try:
            async with self._session_factory() as session:
                row = (
                    await session.execute(
                        update(ToolCacheEntry)
                        .where(
                            ToolCacheEntry.namespace == namespace,
                            ToolCacheEntry.cache_key == cache_key,
                            ToolCacheEntry.schema_version == schema_version,
                            ToolCacheEntry.expires_at > _utcnow(),
                        )
                        .values(
                            last_accessed_at=_utcnow(),
                            hit_count=ToolCacheEntry.hit_count + 1,
                        )
                        .returning(
                            ToolCacheEntry.value_json,
                            ToolCacheEntry.content_hash,
                            ToolCacheEntry.metrics_json,
                        )
                    )
                ).first()
                await session.commit()
            if row is None:
                return None
            return CacheResult(
                value=row.value_json,
                hit=True,
                content_hash=row.content_hash,
                metrics=dict(row.metrics_json or {}),
            )
        except Exception:  # noqa: BLE001 - cache fail-open
            logger.warning(
                "tool_cache_read_failed namespace=%s key=%s",
                namespace,
                cache_key[:12],
                exc_info=True,
            )
            return None

    async def _put(
        self,
        namespace: str,
        cache_key: str,
        value: CacheValue,
        *,
        ttl_seconds: int,
        schema_version: str,
    ) -> bool:
        now = _utcnow()
        values = {
            "namespace": namespace,
            "cache_key": cache_key,
            "value_json": value.value,
            "metrics_json": value.metrics,
            "content_hash": value.content_hash,
            "schema_version": schema_version,
            "created_at": now,
            "expires_at": now + timedelta(seconds=ttl_seconds),
            "last_accessed_at": now,
            "hit_count": 0,
        }
        try:
            async with self._session_factory() as session:
                dialect = session.get_bind().dialect.name
                insert = pg_insert if dialect == "postgresql" else sqlite_insert
                statement = insert(ToolCacheEntry).values(**values)
                statement = statement.on_conflict_do_update(
                    index_elements=["namespace", "cache_key"],
                    set_={
                        key: item
                        for key, item in values.items()
                        if key not in {"namespace", "cache_key"}
                    },
                )
                await session.execute(statement)
                await session.commit()
            return True
        except Exception:  # noqa: BLE001 - cache fail-open
            logger.warning(
                "tool_cache_write_failed namespace=%s key=%s",
                namespace,
                cache_key[:12],
                exc_info=True,
            )
            return False

    async def _join_flight(self, key: tuple[str, str]) -> _Flight:
        async with self._flights_lock:
            flight = self._flights.setdefault(key, _Flight(asyncio.Lock()))
            flight.users += 1
            return flight

    async def _leave_flight(self, key: tuple[str, str], flight: _Flight) -> None:
        async with self._flights_lock:
            flight.users -= 1
            if flight.users == 0 and self._flights.get(key) is flight:
                self._flights.pop(key, None)

    @staticmethod
    def _result(value: CacheValue, *, hit: bool) -> CacheResult:
        return CacheResult(
            value=value.value,
            hit=hit,
            content_hash=value.content_hash,
            metrics=value.metrics,
        )
