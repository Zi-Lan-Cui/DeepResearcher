"""服务层测试的 PostgreSQL 后端：表结构建一次，每个用例开头清空全部表。

必须由 SERVICE_TEST_DATABASE_URL 指定一个可用的库。未配置时相关用例整体跳过，
不退回嵌入式数据库：advisory lock、LISTEN/NOTIFY 与并发写事务在嵌入式库上语义不等价。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

PG_ENV = "SERVICE_TEST_DATABASE_URL"

_schema_ready = False
_reset_marks: set[str] = set()


def postgres_base() -> str | None:
    """返回异步驱动形态的基准 URL；未配置 PostgreSQL 时返回 None。"""
    raw = os.getenv(PG_ENV, "").strip()
    if not raw:
        return None
    if not raw.startswith("postgresql"):
        raise ValueError(f"{PG_ENV} 必须是 PostgreSQL 连接串，实际是 {raw!r}")
    if not raw.startswith("postgresql+asyncpg://"):
        raw = raw.replace("postgresql://", "postgresql+asyncpg://", 1)
    return raw


def database_url_for(tmp_path: Path) -> str:
    """这条用例要连的库。

    本地未配置时跳过整条用例，CI 未配置直接失败。tmp_path 只作为用例身份标记：
    每个测试函数各自唯一，据此决定是否清表。
    """
    base = postgres_base()
    if base is None:
        if os.getenv("CI"):
            raise RuntimeError(f"CI 必须配置 {PG_ENV}，服务层用例不允许静默跳过。")
        pytest.skip(f"需要 PostgreSQL：把 {PG_ENV} 指向一个可用的库")
    _prepare(base, str(tmp_path))
    return base


def _prepare(base: str, marker: str) -> None:
    """首次调用建表，此后每遇到新用例清空一次。"""
    global _schema_ready  # noqa: PLW0603 - 建表只做一次，模块级状态比包一层容器直白
    if not _schema_ready:
        from sqlalchemy import create_engine

        from deepresearcher.service.storage.tables import Base

        engine = create_engine(_engine_url(base))
        try:
            Base.metadata.create_all(engine)
        finally:
            engine.dispose()
        _schema_ready = True
    if marker in _reset_marks:
        return
    _reset_marks.add(marker)
    reset_all(base)


def reset_all(base: str) -> None:
    """清空全部表并重置自增序列，供不走本模块的直连用例自保。"""
    from psycopg import sql

    with _connect(base) as conn:
        rows = conn.execute(
            "select tablename from pg_tables where schemaname = 'public'"
        ).fetchall()
        if not rows:
            return
        names = sorted(row[0] for row in rows)
        conn.execute(
            sql.SQL("TRUNCATE TABLE {} RESTART IDENTITY CASCADE").format(
                sql.SQL(", ").join(sql.Identifier(name) for name in names)
            )
        )


def _connect(base: str):
    import psycopg

    conn = psycopg.connect(_dsn(base), autocommit=True)
    # 上一用例若漏了连接，TRUNCATE 会一直等锁；宁可快速失败也不挂住整个会话。
    conn.execute("SET lock_timeout = '10s'")
    return conn


def _engine_url(base: str) -> str:
    """SQLAlchemy 用的 URL，驱动取 psycopg 以复用同一份依赖。"""
    return base.replace("+asyncpg", "+psycopg", 1)


def _dsn(base: str) -> str:
    """psycopg 直连用的 DSN；libpq 不认识 SQLAlchemy 的 +driver 后缀。"""
    return base.replace("+asyncpg", "", 1)
