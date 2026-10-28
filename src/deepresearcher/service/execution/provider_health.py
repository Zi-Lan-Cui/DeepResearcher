"""记录搜索提供方健康状态"""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from deepresearcher.service.storage.tables import ProviderHealthRecord
from deepresearcher.service.storage.tables import utcnow as _utcnow


class ProviderHealthStore:
    def __init__(self, session_factory: Callable[[], Any]) -> None:
        self._session_factory = session_factory

    async def is_open(self, provider: str) -> str | None:
        async with self._session_factory() as session:
            row = await session.scalar(
                select(ProviderHealthRecord).where(ProviderHealthRecord.provider == provider)
            )
        if row is None:
            return None
        return row.reason if row.open_until > _utcnow() else None

    async def trip(self, provider: str, reason: str, seconds: float) -> None:
        for attempt in range(2):
            async with self._session_factory() as session:
                now = _utcnow()
                until = now + timedelta(seconds=seconds)
                row = await session.scalar(
                    select(ProviderHealthRecord).where(ProviderHealthRecord.provider == provider)
                )
                if row is None:
                    session.add(
                        ProviderHealthRecord(
                            provider=provider, reason=reason, open_until=until, updated_at=now
                        )
                    )
                else:
                    # 只延后、不缩短：并发多 worker 撞同一 key 时保留最大恢复窗口。
                    if until > row.open_until:
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
