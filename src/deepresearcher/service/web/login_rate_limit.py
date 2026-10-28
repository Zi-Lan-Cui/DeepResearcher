"""所有 API 进程共享的、由数据库支撑的登录限速。"""

from __future__ import annotations

import hashlib
import hmac
import math
from dataclasses import dataclass
from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from deepresearcher.service.coordination.locks import acquire_xact_lock
from deepresearcher.service.storage.tables import LoginThrottle
from deepresearcher.service.storage.tables import utcnow as _utcnow


@dataclass(frozen=True)
class LoginRateLimitResult:
    allowed: bool
    retry_after_seconds: int = 0


class LoginRateLimiter:
    """跨 API 进程原子地消耗登录预算。

    事务级 advisory lock 按隐私安全键串行化，跨 API 进程原子生效。
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        secret: str,
        account_attempts: int,
        ip_attempts: int,
        window_seconds: int,
        block_seconds: int,
    ) -> None:
        self._session_factory = session_factory
        self._secret = secret.encode()
        self._account_attempts = account_attempts
        self._ip_attempts = ip_attempts
        self._window = timedelta(seconds=window_seconds)
        self._block = timedelta(seconds=block_seconds)

    async def consume(self, *, client_ip: str, email: str) -> LoginRateLimitResult:
        # 获取ip和账号限制
        budgets = (
            (self._key("ip", client_ip), self._ip_attempts),
            (self._key("account", email), self._account_attempts),
        )
        async with self._session_factory() as session:
            async with session.begin():
                for key_hash, _limit in sorted(budgets):
                    await acquire_xact_lock(session, self._lock_id(key_hash))
                return await self._consume_locked(session, budgets)

    async def clear_account(self, email: str) -> None:
        key_hash = self._key("account", email)
        async with self._session_factory() as session:
            async with session.begin():
                await acquire_xact_lock(session, self._lock_id(key_hash))
                row = await session.get(LoginThrottle, key_hash)
                if row is not None:
                    await session.delete(row)

    async def _consume_locked(
        self,
        session: AsyncSession,
        budgets: tuple[tuple[str, int], tuple[str, int]],
    ) -> LoginRateLimitResult:
        now = _utcnow()
        rows = {
            row.key_hash: row
            for row in (
                await session.scalars(
                    select(LoginThrottle).where(
                        LoginThrottle.key_hash.in_(key for key, _limit in budgets)
                    )
                )
            ).all()
        }
        retry_after = 0
        for key_hash, limit in budgets:
            row = rows.get(key_hash)
            if row is None:
                continue
            if row.blocked_until is not None and row.blocked_until > now:
                retry_after = max(
                    retry_after,
                    math.ceil((row.blocked_until - now).total_seconds()),
                )
            # IP 和 账号均被封禁，取最长封禁时间
            elif now - row.window_started_at < self._window and row.attempt_count >= limit:
                row.blocked_until = now + self._block
                row.updated_at = now
                retry_after = max(retry_after, math.ceil(self._block.total_seconds()))
        if retry_after:
            return LoginRateLimitResult(False, retry_after)

        for key_hash, _limit in budgets:
            row = rows.get(key_hash)
            # 首次尝试或窗口过期，重置计数器；否则累加计数器。
            if row is None:
                session.add(
                    LoginThrottle(
                        key_hash=key_hash,
                        attempt_count=1,
                        window_started_at=now,
                        updated_at=now,
                    )
                )
            elif now - row.window_started_at >= self._window:
                row.attempt_count = 1
                row.window_started_at = now
                row.blocked_until = None
                row.updated_at = now
            else:
                row.attempt_count += 1
                row.updated_at = now
        return LoginRateLimitResult(True)

    def _key(self, dimension: str, value: str) -> str:
        return hmac.new(
            self._secret,
            f"login:{dimension}:{value}".encode(),
            hashlib.sha256,
        ).hexdigest()

    @staticmethod
    def _lock_id(key_hash: str) -> int:
        return int.from_bytes(bytes.fromhex(key_hash[:16]), signed=True)
