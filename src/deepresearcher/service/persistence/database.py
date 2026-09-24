"""异步数据库引擎与会话工厂。

支持两种 URL：``postgresql+asyncpg://...``（服务）与 ``sqlite+aiosqlite://...``
（测试/无 Docker 过渡）。``:memory:`` 的 SQLite 每个连接是独立库，必须
StaticPool 复用同一连接；外键（CASCADE）在 SQLite 里默认关闭，需逐连接开 pragma。

应用启动用 ``migrate_database`` 以 ``create_all``(幂等)确保 schema 存在；
``init_db`` 是同一套 metadata 的隔离测试入口。个人项目、未上线，不维护
Alembic 版本迁移链——schema 演进直接改 models 再重建即可。
"""

from __future__ import annotations

from sqlalchemy import event, text
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
    """为隔离的 SQLite 测试创建完整 schema。"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def migrate_database(database_url: str) -> None:
    """幂等地确保应用 schema 存在（创建缺失的表）。

    本项目未上线，不维护 Alembic 迁移链：schema 由模型元数据经
    ``create_all``（checkfirst）推出，重跑是空操作，加表只改模型。
    API 与多个 Worker 可能同时启动，因此 PostgreSQL 上把 DDL 包在
    事务级 advisory lock 里串行化并发 create_all；SQLite 是单进程测试路径。
    """
    engine = make_engine(database_url)
    try:
        async with engine.begin() as connection:
            if connection.dialect.name == "postgresql":
                # 事务级 advisory lock 在提交时自动释放，
                # 保护下方并发的跨进程 DDL。
                await connection.execute(
                    text("SELECT pg_advisory_xact_lock(:lock_id)"),
                    {"lock_id": DATABASE_MIGRATION_LOCK_ID},
                )
            await connection.run_sync(Base.metadata.create_all)
    finally:
        await engine.dispose()
