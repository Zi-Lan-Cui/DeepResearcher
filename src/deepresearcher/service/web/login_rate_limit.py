"""Database-backed login throttling shared by every API process."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from deepresearcher.service.persistence.models import LoginThrottle
from deepresearcher.service.persistence.models import utcnow as _utcnow


def _aware(value: datetime) -> datetime:
    # SQLite drops timezone information; production PostgreSQL preserves it.
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class LoginRateLimitResult:
    allowed: bool
    retry_after_seconds: int = 0


class LoginRateLimiter:
    """Consume login budgets atomically across API processes.

    PostgreSQL advisory transaction locks serialize each privacy-preserving key.
    The asyncio lock is only the SQLite/test compatibility path.
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
        self._sqlite_lock = asyncio.Lock()

    async def consume(self, *, client_ip: str, email: str) -> LoginRateLimitResult:
        budgets = (
            (self._key("ip", client_ip), self._ip_attempts),
            (self._key("account", email), self._account_attempts),
        )
        async with self._session_factory() as session:
            is_postgres = session.get_bind().dialect.name == "postgresql"
            lock = _NullAsyncContext() if is_postgres else self._sqlite_lock
            async with lock:
                async with session.begin():
                    if is_postgres:
                        for key_hash, _limit in sorted(budgets):
                            await session.execute(
                                text("SELECT pg_advisory_xact_lock(:lock_id)"),
                                {"lock_id": self._lock_id(key_hash)},
                            )
                    return await self._consume_locked(session, budgets)

    async def clear_account(self, email: str) -> None:
        key_hash = self._key("account", email)
        async with self._session_factory() as session:
            is_postgres = session.get_bind().dialect.name == "postgresql"
            lock = _NullAsyncContext() if is_postgres else self._sqlite_lock
            async with lock:
                async with session.begin():
                    if is_postgres:
                        await session.execute(
                            text("SELECT pg_advisory_xact_lock(:lock_id)"),
                            {"lock_id": self._lock_id(key_hash)},
                        )
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
            if row.blocked_until is not None and _aware(row.blocked_until) > now:
                retry_after = max(
                    retry_after,
                    math.ceil((_aware(row.blocked_until) - now).total_seconds()),
                )
            elif now - _aware(row.window_started_at) < self._window and row.attempt_count >= limit:
                row.blocked_until = now + self._block
                row.updated_at = now
                retry_after = max(retry_after, math.ceil(self._block.total_seconds()))
        if retry_after:
            return LoginRateLimitResult(False, retry_after)

        for key_hash, _limit in budgets:
            row = rows.get(key_hash)
            if row is None:
                session.add(
                    LoginThrottle(
                        key_hash=key_hash,
                        attempt_count=1,
                        window_started_at=now,
                        updated_at=now,
                    )
                )
            elif now - _aware(row.window_started_at) >= self._window:
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


class _NullAsyncContext:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *_args: object) -> None:
        return None
