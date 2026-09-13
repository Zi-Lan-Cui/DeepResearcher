"""异步数据库引擎与会话工厂。

支持两种 URL：``postgresql+asyncpg://...``（服务）与 ``sqlite+aiosqlite://...``
（测试/无 Docker 过渡）。``:memory:`` 的 SQLite 每个连接是独立库，必须
StaticPool 复用同一连接；外键（CASCADE）在 SQLite 里默认关闭，需逐连接开 pragma。

应用启动使用 Alembic 升级 schema；``init_db`` 只供隔离测试快速创建当前完整模型。
首个迁移前已经存在的数据库会被采纳到 ``0001_initial``，再执行 lease 字段迁移。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import event, inspect, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool, StaticPool

from deepresearcher.service.coordination import DATABASE_MIGRATION_LOCK_ID
from deepresearcher.service.persistence.models import Base


def make_engine(database_url: str) -> AsyncEngine:
    kwargs: dict = {"echo": False}
    if database_url.startswith("sqlite"):
        if ":memory:" in database_url:
            kwargs["poolclass"] = StaticPool
        else:
            # 文件库用 NullPool：连接随 session 归还即关（await 完成），
            # 不依赖 dispose 回收池——dispose 之后 aiosqlite 工作线程的迟到
            # 回调会在已关闭循环上 call_soon_threadsafe（测试里表现为归因到
            # 后续用例的 UnhandledThreadException 竞态告警）。
            kwargs["poolclass"] = NullPool
        if "aiosqlite" in database_url:
            kwargs["connect_args"] = {"check_same_thread": False}
    engine = create_async_engine(database_url, **kwargs)
    if engine.dialect.name == "sqlite":
        _configure_sqlite(engine)
    return engine


def _configure_sqlite(engine: AsyncEngine) -> None:
    """逐连接会话层前置条件，让测试库具备生产 asyncpg 的并发语义：

    - foreign_keys：SQLite 默认关闭，不开则 ON DELETE CASCADE 静默失效；
    - busy_timeout：读一写一并发时（RunExecutor 后台 flush vs 请求事务）默认
      立刻抛 database is locked，5s 等待等价于 PG 的行锁排队；
    - journal_mode=WAL：读写不互斥（仅文件库有效，:memory: 无副作用）。
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
    # expire_on_commit=False：提交后的实例属性仍可读，避免异步下惰性刷新炸线程。
    return async_sessionmaker(engine, expire_on_commit=False)


async def init_db(engine: AsyncEngine) -> None:
    """Create the complete schema for isolated SQLite tests."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def migrate_database(database_url: str) -> None:
    """Upgrade an application database with Alembic.

    Databases created before Alembic are adopted at the immutable initial-schema
    revision and then upgraded. This is safe only because ``0001_initial`` exactly
    describes the schema that preceded the first ALTER migration.
    """
    # Do not couple migration discovery to this module's package depth: the
    # persistence layer can move without silently pointing Alembic at ``src/``.
    root = next(
        parent for parent in Path(__file__).resolve().parents if (parent / "alembic.ini").is_file()
    )
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "migrations"))
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))

    def upgrade(tables: set[str]) -> None:
        if "runs" in tables and "alembic_version" not in tables:
            command.stamp(config, "0001_initial")
        command.upgrade(config, "head")

    # API 与多个 Worker 可能同时启动。Alembic 自身不会为多进程
    # upgrade 排队，因此在 PostgreSQL 上用 session advisory lock 包住整段
    # inspect + upgrade；SQLite 只是单进程测试路径。
    engine = make_engine(database_url)
    try:
        async with engine.connect() as connection:
            is_postgres = connection.dialect.name == "postgresql"
            if is_postgres:
                await connection.execute(
                    text("SELECT pg_advisory_lock(:lock_id)"),
                    {"lock_id": DATABASE_MIGRATION_LOCK_ID},
                )
            try:
                tables = await connection.run_sync(
                    lambda sync: set(inspect(sync).get_table_names())
                )
                await asyncio.to_thread(upgrade, tables)
            finally:
                if is_postgres:
                    await connection.execute(
                        text("SELECT pg_advisory_unlock(:lock_id)"),
                        {"lock_id": DATABASE_MIGRATION_LOCK_ID},
                    )
    finally:
        await engine.dispose()
