"""异步数据库引擎与会话工厂。

支持 postgresql+asyncpg:// 与 sqlite+aiosqlite://。SQLite 的 :memory: 每个连接
是独立库，必须用 StaticPool 复用；外键 CASCADE 默认关闭，需逐连接开 pragma。
启动用 migrate_database 以幂等 create_all 建表，仓库不维护 Alembic 迁移链。
"""

from __future__ import annotations

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from deepresearcher.service.coordination import DATABASE_MIGRATION_LOCK_ID
from deepresearcher.service.persistence.advisory_lock import acquire_xact_lock
from deepresearcher.service.persistence.models import Base


def make_engine(database_url: str) -> AsyncEngine:
    kwargs: dict = {"echo": False}
    if database_url.startswith("sqlite"):
        # 文件库用默认连接池。NullPool 每条 SQL 新建连接，并发时两个 aiosqlite
        # 线程同时首开同一 WAL 库，Windows 上会返回 readonly。
        if ":memory:" in database_url:
            kwargs["poolclass"] = StaticPool
        if "aiosqlite" in database_url:
            kwargs["connect_args"] = {"check_same_thread": False}
    engine = create_async_engine(database_url, **kwargs)
    if engine.dialect.name == "sqlite":
        _configure_sqlite(engine)
    return engine


def _configure_sqlite(engine: AsyncEngine) -> None:
    """逐连接会话层前置条件，让测试库具备生产 asyncpg 的并发语义。

    foreign_keys 不开则 ON DELETE CASCADE 静默失效；busy_timeout 避免并发时立刻抛
    database is locked；journal_mode=WAL 让读写不互斥（仅文件库有效）。
    """

    @event.listens_for(engine.sync_engine, "connect")
    def _pragmas_on_connect(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.execute("PRAGMA journal_mode=WAL")
        finally:
            cursor.close()


def make_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    # expire_on_commit=False：提交后实例属性仍可读，避免异步会话在访问过期
    # 属性时触发惰性刷新而出错。
    return async_sessionmaker(engine, expire_on_commit=False)


async def init_db(engine: AsyncEngine) -> None:
    """为隔离的 SQLite 测试创建完整 schema。"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def migrate_database(database_url: str) -> None:
    """幂等地确保应用 schema 存在（创建缺失的表）。

    schema 由模型元数据经 create_all 推出，仓库不维护 Alembic 迁移链。API 与多个 Worker
    可能同时启动，PostgreSQL 上把 DDL 放进事务级 advisory lock 串行化。
    """
    engine = make_engine(database_url)
    try:
        async with engine.begin() as connection:
            # 事务级 advisory lock 在提交时自动释放，保护下方并发的跨进程 DDL;
            # SQLite 单进程测试路径上空操作。
            await acquire_xact_lock(connection, DATABASE_MIGRATION_LOCK_ID)
            await connection.run_sync(Base.metadata.create_all)
    finally:
        await engine.dispose()
