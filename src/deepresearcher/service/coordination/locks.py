"""PostgreSQL advisory lock

两种锁生命周期不同，不可选错：acquire_xact_lock 事务级，commit/rollback 自动释放；
held_session_lock 与 held_autocommit_lock 会话级，跨代码块持有、退出时显式释放，
进程崩溃随连接断开自动解锁。后两者的区别只是客户端：前者接 SQLAlchemy 会话，
后者接 psycopg 连接。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import text

_XACT_LOCK = text("SELECT pg_advisory_xact_lock(:lock_id)")
_XACT_TRY_LOCK = text("SELECT pg_try_advisory_xact_lock(:lock_id)")
_SESSION_LOCK = text("SELECT pg_advisory_lock(:lock_id)")
_SESSION_UNLOCK = text("SELECT pg_advisory_unlock(:lock_id)")
_PSYCOPG_SESSION_LOCK = "SELECT pg_advisory_lock(%s)"
_PSYCOPG_SESSION_UNLOCK = "SELECT pg_advisory_unlock(%s)"


async def acquire_xact_lock(session_or_connection: Any, key: int) -> None:
    """取事务级 advisory 锁，commit/rollback 时自动释放。"""
    await session_or_connection.execute(_XACT_LOCK, {"lock_id": key})


async def try_acquire_xact_lock(session_or_connection: Any, key: int) -> bool:
    """非阻塞事务级 advisory 锁：抢到 True，已被占 False。

    周期清扫用——取不到锁的副本本轮直接跳过，不与持有者排队，
    否则 N 个 worker 会在锁上串成一条比清扫本身更长的队。
    """
    result = await session_or_connection.execute(_XACT_TRY_LOCK, {"lock_id": key})
    return bool(result.scalar())


@asynccontextmanager
async def held_session_lock(session: Any, key: int) -> AsyncIterator[None]:
    """会话级 advisory 锁的上下文管理器：进块前取锁，退出时释放。"""
    await session.execute(_SESSION_LOCK, {"lock_id": key})
    try:
        yield
    finally:
        await session.execute(_SESSION_UNLOCK, {"lock_id": key})


@asynccontextmanager
async def held_autocommit_lock(connection: Any, key: int) -> AsyncIterator[None]:
    """会话级 advisory 锁，接 psycopg 连接：进块前取锁，退出时释放。

    调用方的连接必须 autocommit：被保护的 DDL 可能含 CREATE INDEX CONCURRENTLY，
    它要等所有快照更早的事务结束，锁放在未提交事务里就是自己等自己。
    """
    async with connection.cursor() as cursor:
        await cursor.execute(_PSYCOPG_SESSION_LOCK, (key,))
    try:
        yield
    finally:
        async with connection.cursor() as cursor:
            await cursor.execute(_PSYCOPG_SESSION_UNLOCK, (key,))
