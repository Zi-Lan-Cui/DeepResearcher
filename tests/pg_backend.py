"""服务层测试的数据库后端选择。

默认用 tmp_path 下的 SQLite 文件库，不依赖外部服务。设置 SERVICE_TEST_DATABASE_URL
后改用该 PostgreSQL：表结构建一次，每条用例开头清空全部表——SQLite 靠一文件一库
拿到的隔离度，在共享库上只能这么还。
"""

from __future__ import annotations

import os
from pathlib import Path

PG_ENV = "SERVICE_TEST_DATABASE_URL"

_schema_ready = False
_reset_marks: set[str] = set()


def postgres_base() -> str | None:
    """返回异步驱动形态的基准 URL；未配置 PostgreSQL 时返回 None。"""
    raw = os.getenv(PG_ENV, "").strip()
    if not raw.startswith("postgresql"):
        return None
    if not raw.startswith("postgresql+asyncpg://"):
        raw = raw.replace("postgresql://", "postgresql+asyncpg://", 1)
    return raw


def database_url_for(tmp_path: Path, label: str) -> str:
    """这条用例要连的库：未配置 PG 时是独占的 SQLite 文件，否则是清空后的共享 PG 库。"""
    base = postgres_base()
    if base is None:
        return f"sqlite+aiosqlite:///{tmp_path / (label + '.db')}"
    _prepare(base, str(tmp_path))
    return base


def _prepare(base: str, marker: str) -> None:
    """首次调用建表，此后每遇到新用例清空一次。"""
    global _schema_ready
    if not _schema_ready:
        from sqlalchemy import create_engine

        from deepresearcher.service.persistence.models import Base

        engine = create_engine(_engine_url(base))
        try:
            Base.metadata.create_all(engine)
        finally:
            engine.dispose()
        _schema_ready = True
    if marker in _reset_marks:
        return
    _reset_marks.add(marker)
    _truncate_all(base)


def _truncate_all(base: str) -> None:
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
    """SQLAlchemy 用的 URL，驱动必须是 psycopg 才能顺手复用同一份依赖。"""
    return base.replace("+asyncpg", "+psycopg", 1)


def _dsn(base: str) -> str:
    """psycopg 直连用的 DSN；libpq 不认识 SQLAlchemy 的 +driver 后缀。"""
    return base.replace("+asyncpg", "", 1)
