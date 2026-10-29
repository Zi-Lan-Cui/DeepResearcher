"""搜索提供方健康状态的持久化账本。"""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError

from deepresearcher.health import Cooldown, Unavailable
from deepresearcher.service.storage.tables import ProviderHealthRecord
from deepresearcher.service.storage.tables import utcnow as _utcnow


class ProviderHealthStore:
    """跨 worker 共享的冷却位；语义与 MemoryHealthLedger 一致。"""

    def __init__(self, session_factory: Callable[[], Any]) -> None:
        self._session_factory = session_factory

    async def unavailable(self, key: str) -> Unavailable | None:
        async with self._session_factory() as session:
            row = await session.scalar(
                select(ProviderHealthRecord).where(ProviderHealthRecord.provider == key)
            )
        if row is None:
            return None
        remaining = (row.open_until - _utcnow()).total_seconds()
        return Unavailable(row.reason, remaining) if remaining > 0 else None

    async def record_failure(self, key: str, reason: str, policy: Cooldown) -> None:
        for attempt in range(2):
            try:
                await self._record_failure_once(key, reason, policy)
            except IntegrityError:
                # 并发首次插入撞主键：重来一次即可在已有行上递增。
                if attempt == 1:
                    raise
                continue
            return

    async def _record_failure_once(self, key: str, reason: str, policy: Cooldown) -> None:
        async with self._session_factory() as session:
            now = _utcnow()
            row = await session.scalar(
                select(ProviderHealthRecord)
                .where(ProviderHealthRecord.provider == key)
                .with_for_update()
            )
            if row is None:
                row = ProviderHealthRecord(
                    provider=key, reason=reason, open_until=now, updated_at=now
                )
                session.add(row)
                await session.flush()
            row.failures += 1
            row.reason = reason
            row.updated_at = now
            if row.failures >= policy.threshold:
                row.failures = 0
                row.strikes += 1
                row.open_until = now + timedelta(seconds=policy.delay_for(row.strikes))
            await session.commit()

    async def record_success(self, key: str) -> None:
        async with self._session_factory() as session:
            await session.execute(
                delete(ProviderHealthRecord).where(ProviderHealthRecord.provider == key)
            )
            await session.commit()

    async def trip(self, key: str, reason: str, seconds: float) -> None:
        """标记不可用；已有更长窗口时不缩短它。trip 不计 strikes。"""
        for attempt in range(2):
            async with self._session_factory() as session:
                now = _utcnow()
                until = now + timedelta(seconds=seconds)
                row = await session.scalar(
                    select(ProviderHealthRecord).where(ProviderHealthRecord.provider == key)
                )
                if row is None:
                    session.add(
                        ProviderHealthRecord(
                            provider=key, reason=reason, open_until=until, updated_at=now
                        )
                    )
                elif until > row.open_until:
                    row.reason = reason
                    row.open_until = until
                    row.updated_at = now
                try:
                    await session.commit()
                except IntegrityError:
                    await session.rollback()
                    if attempt == 1:
                        raise
                    continue
                return
