"""PostgreSQL advisory lock 原语:全仓唯一直写锁 SQL 的地方。

advisory lock 是对话连接的能力,不是任何一张表的领域;各平面借它串行化
"计数+写"这类跨行不变量。非 PG(SQLite 测试)统一在此降级为空操作,
调用点不再各自写方言判断。两种锁的生命周期不同,选错就是 bug:

- acquire_xact_lock:事务级,commit/rollback 自动释放,罩住当前事务的临界区;
- held_session_lock:会话级,跨任意代码块持有、退出时显式释放;进程崩溃时
  随连接断开自动解锁,适合"启动归类跑完才放行"这类跨事务的编排。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import text

_XACT_LOCK = text("SELECT pg_advisory_xact_lock(:lock_id)")
_SESSION_LOCK = text("SELECT pg_advisory_lock(:lock_id)")
_SESSION_UNLOCK = text("SELECT pg_advisory_unlock(:lock_id)")


def _is_postgres(session_or_connection: Any) -> bool:
    # AsyncConnection 直接暴露 .dialect;AsyncSession 只有 get_bind()。两条路都走。
    dialect = getattr(session_or_connection, "dialect", None)
    if dialect is None:
        dialect = session_or_connection.get_bind().dialect
    return dialect.name == "postgresql"


async def acquire_xact_lock(session_or_connection: Any, key: int) -> None:
    """取事务级 advisory 锁;非 PG 上空操作(各调用点的进程内锁负责降级)。"""
    if _is_postgres(session_or_connection):
        await session_or_connection.execute(_XACT_LOCK, {"lock_id": key})


@asynccontextmanager
async def held_session_lock(session: Any, key: int) -> AsyncIterator[None]:
    """会话级 advisory 锁的上下文:先进块后解锁;非 PG 直接穿过。"""
    if not _is_postgres(session):
        yield
        return
    await session.execute(_SESSION_LOCK, {"lock_id": key})
    try:
        yield
    finally:
        await session.execute(_SESSION_UNLOCK, {"lock_id": key})
