"""异步数据库引擎与会话工厂。"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from deepresearcher.service.coordination.keys import DATABASE_MIGRATION_LOCK_ID
from deepresearcher.service.coordination.locks import acquire_xact_lock
from deepresearcher.service.storage.tables import Base


def make_engine(database_url: str) -> AsyncEngine:
    return create_async_engine(database_url, echo=False)


def make_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    # expire_on_commit=False：提交后实例属性仍可读，避免异步会话在访问过期
    # 属性时触发惰性刷新而出错。
    return async_sessionmaker(engine, expire_on_commit=False)


async def migrate_database(database_url: str) -> None:
    """幂等地确保应用 schema 存在（创建缺失的表）。

    schema 由模型元数据经 create_all 推出。API 与多个 Worker 可能同时启动，
    把 DDL 放进事务级 advisory lock 串行化。
    """
    engine = make_engine(database_url)
    try:
        async with engine.begin() as connection:
            # 事务级 advisory lock 在提交时自动释放，保护下方并发的跨进程 DDL。
            await acquire_xact_lock(connection, DATABASE_MIGRATION_LOCK_ID)
            await connection.run_sync(Base.metadata.create_all)
    finally:
        await engine.dispose()
