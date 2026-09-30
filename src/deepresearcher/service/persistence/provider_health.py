"""跨 worker 共享的搜索提供方健康位（Postgres 实现）。

`tools/web/search/health.py` 的内存版只在单进程生效；多 worker 部署下，一个 worker
撞到额度/鉴权耗尽，其它 worker 仍会各自出网撞。本实现把熔断位落到 PostgreSQL，
遇到额度错误时，先发现者写熔断位，其余 worker 在恢复窗口内快速失败；熔断不停止进程。
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from deepresearcher.service.persistence.models import ProviderHealthRecord, as_utc
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
        # SQLite 回读 naive、asyncpg aware：as_utc 统一口径再比较。
        return row.reason if as_utc(row.open_until) > _utcnow() else None

    async def trip(self, provider: str, reason: str, seconds: float) -> None:
        # select-then-insert 在多 worker 并发首次写入同一 key 时，后提交方会收到
        # IntegrityError；调用点处于 except ProviderExhaustedError 分支内，
        # 任何熔断写入都会当场替换原始业务异常类型。冲突即重走 update 分支重试。
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
                    if until > as_utc(row.open_until):
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
