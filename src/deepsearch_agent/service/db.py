"""异步数据库引擎与会话工厂。

支持两种 URL：``postgresql+asyncpg://...``（服务）与 ``sqlite+aiosqlite://...``
（测试/无 Docker 过渡）。``:memory:`` 的 SQLite 每个连接是独立库，必须
StaticPool 复用同一连接；外键（CASCADE）在 SQLite 里默认关闭，需逐连接开 pragma。

迁移策略：P0 用 ``init_db``（create_all）——仅当 schema 保持 append-only 且部署
只有我们一个时成立。**引入 Alembic 的触发条件是第一次 ALTER/列变更，不是第一次
上线**；到那步之前禁止手改 models 里已有列的语义。
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

from deepsearch_agent.service.models import Base


def make_engine(database_url: str) -> AsyncEngine:
    kwargs: dict = {"echo": False}
    if database_url.startswith("sqlite"):
        if ":memory:" in database_url:
            kwargs["poolclass"] = StaticPool
        if "aiosqlite" in database_url:
            kwargs["connect_args"] = {"check_same_thread": False}
    engine = create_async_engine(database_url, **kwargs)
    if engine.dialect.name == "sqlite":
        _enable_sqlite_foreign_keys(engine)
    return engine


def _enable_sqlite_foreign_keys(engine: AsyncEngine) -> None:
    @event.listens_for(engine.sync_engine, "connect")
    def _fk_pragma_on_connect(dbapi_connection, _connection_record):  # pragma: no cover
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
        finally:
            cursor.close()


def make_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    # expire_on_commit=False：提交后的实例属性仍可读，避免异步下惰性刷新炸线程。
    return async_sessionmaker(engine, expire_on_commit=False)


async def init_db(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
