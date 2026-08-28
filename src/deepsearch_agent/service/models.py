"""SQLAlchemy ORM 模型：users / runs / run_events。

字段类型刻意保持方言中立——只用通用 ``JSON``（PG 与 SQLite 都映射为 JSON），
因此测试可用 ``sqlite+aiosqlite`` 内存库与生产 ``postgresql+asyncpg`` 跑同一套模型。
不要在这里引入 JSONB、``server_default`` 等只属于单一方言的构造。
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# 运行状态取值。P0 用普通 str 而非枚举/CHECK 约束，便于增删而不触发 ALTER；
# 合法取值集合在 service/runs.py 里由 RunManager 单点维护。
RUN_STATUSES = ("queued", "running", "completed", "failed", "cancelled")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    runs: Mapped[list["Run"]] = relationship(back_populates="user")


class Run(Base):
    __tablename__ = "runs"

    # 文本主键 "run-<hex>"：与引擎 run_id 同源，避免自增 id 泄露运行总量。
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    query: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), default="queued")
    answer_mode: Mapped[str | None] = mapped_column(String(32))
    terminal_reason: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)
    report_markdown: Mapped[str | None] = mapped_column(Text)
    citations_json: Mapped[list | None] = mapped_column(JSON)
    evidence_count: Mapped[int] = mapped_column(Integer, default=0)
    source_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    user: Mapped["User"] = relationship(back_populates="runs")
    events: Mapped[list["RunEvent"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", passive_deletes=True
    )


class RunEvent(Base):
    """研究过程事件的留档，供 SSE 断线重连回放与审计。

    ``record`` 存引擎写出的原始事件 dict——引擎侧已对所有 ``*_preview``/error 做了
    截断（≤1000 字），故单行体积天然有界。投影（面向用户的文案）不落库，回放时
    经 projector 现算，保证「一个投影器 = 一份真相」。
    """

    __tablename__ = "run_events"

    run_id: Mapped[str] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), primary_key=True
    )
    seq: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_type: Mapped[str] = mapped_column(String(64))
    record: Mapped[dict] = mapped_column(JSON)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    run: Mapped["Run"] = relationship(back_populates="events")
