"""跨 worker 共享的搜索提供方健康位（Postgres 实现）。

`tools/web/search/health.py` 的内存版只在单进程生效；多 worker 部署下，一个 worker
撞到额度/鉴权耗尽，其它 worker 仍会各自出网撞。本实现把熔断位落到 PostgreSQL，
谁先撞谁写、其余 worker 在窗口内直接快速失败——"一处发现、全体停手"，但**不停进程**。
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from deepresearcher.service.persistence.models import ProviderHealthRecord
from deepresearcher.service.persistence.models import utcnow as _utcnow


class PostgresProviderHealth:
    def __init__(self, session_factory: Callable[[], Any]) -> None:
        self._session_factory = session_factory

    async def is_open(self, provider: str) -> str | None:
        async with self._session_factory() as session:
            row = await session.scalar(
                select(ProviderHealthRecord).where(ProviderHealthRecord.provider == provider)
            )
        if row is None:
            return None
        until = row.open_until
        # SQLite 回读为 naive datetime，asyncpg 为 aware；统一按 UTC 解释再比较。
        if until.tzinfo is None:
            until = until.replace(tzinfo=timezone.utc)
        return row.reason if until > _utcnow() else None

    async def trip(self, provider: str, reason: str, seconds: float) -> None:
        # select-then-insert 在多 worker 并发首撞同一 key 时会让后提交方吃
        # IntegrityError；调用点住在 except ProviderExhaustedError 分支内，
        # 任熔断开闸当场替换原始业务异常类型。冲突即重走 update 分支重试。
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
                    existing = row.open_until
                    if existing.tzinfo is None:
                        existing = existing.replace(tzinfo=timezone.utc)
                    if until > existing:
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
