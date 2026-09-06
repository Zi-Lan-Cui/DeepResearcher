"""SQLAlchemy ORM 模型：users / runs / run_events。

字段类型刻意保持方言中立——只用通用 ``JSON``（PG 与 SQLite 都映射为 JSON），
因此测试可用 ``sqlite+aiosqlite`` 内存库与生产 ``postgresql+asyncpg`` 跑同一套模型。
不要在这里引入 JSONB、``server_default`` 等只属于单一方言的构造。
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, Numeric, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# 运行状态取值。P0 用普通 str 而非枚举/CHECK 约束，便于增删而不触发 ALTER；
# 该常量是 ORM 层的单一合法集合，控制面与执行面共用同一词汇。
RUN_STATUSES = (
    "queued",
    "running",
    "interrupted",
    "awaiting_input",
    "completed",
    "failed",
    "cancelled",
)


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


class LoginThrottle(Base):
    """Shared login-attempt window; keys are HMACs, never raw email/IP values."""

    __tablename__ = "login_throttles"

    key_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    window_started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    blocked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


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
    lease_owner: Mapped[str | None] = mapped_column(String(96), index=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    attempt: Mapped[int] = mapped_column(Integer, default=0)
    cancellation_requested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    resume_payload: Mapped[dict | None] = mapped_column(JSON)
    event_seq: Mapped[int] = mapped_column(Integer, default=0)
    llm_call_count: Mapped[int] = mapped_column(Integer, default=0)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cached_input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    external_request_count: Mapped[int] = mapped_column(Integer, default=0)
    peak_llm_concurrency: Mapped[int] = mapped_column(Integer, default=0)
    estimated_cost_usd: Mapped[float] = mapped_column(Numeric(14, 8), default=0)
    cache_hit_count: Mapped[int] = mapped_column(Integer, default=0)
    saved_external_request_count: Mapped[int] = mapped_column(Integer, default=0)
    saved_llm_call_count: Mapped[int] = mapped_column(Integer, default=0)
    saved_tokens: Mapped[int] = mapped_column(Integer, default=0)
    saved_cost_usd: Mapped[float] = mapped_column(Numeric(14, 8), default=0)

    user: Mapped["User"] = relationship(back_populates="runs")
    events: Mapped[list["RunEvent"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", passive_deletes=True
    )
    usage_records: Mapped[list["RunUsage"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", passive_deletes=True
    )


class RunEvent(Base):
    """研究过程事件的留档，供 SSE 断线重连回放与审计。

    ``record`` 存引擎写出的原始事件 dict——引擎侧已对所有 ``*_preview``/error 做了
    截断（≤1000 字），故单行体积天然有界。投影（面向用户的文案）不落库，回放时
    经 projector 现算，保证「一个投影器 = 一份真相」。
    """

    __tablename__ = "run_events"

    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), primary_key=True)
    seq: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_type: Mapped[str] = mapped_column(String(64))
    record: Mapped[dict] = mapped_column(JSON)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    run: Mapped["Run"] = relationship(back_populates="events")


class RunUsage(Base):
    """One billable or externally capacity-consuming operation."""

    __tablename__ = "run_usage"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    category: Mapped[str] = mapped_column(String(32), index=True)
    component: Mapped[str] = mapped_column(String(64))
    provider: Mapped[str | None] = mapped_column(String(64))
    model: Mapped[str | None] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(16))
    usage_estimated: Mapped[bool] = mapped_column(Boolean, default=False)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cached_input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Numeric(14, 8), default=0)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    detail_json: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    run: Mapped["Run"] = relationship(back_populates="usage_records")


class ToolCacheEntry(Base):
    """与 run/task 身份无关的工具成功结果。"""

    __tablename__ = "tool_cache_entries"

    namespace: Mapped[str] = mapped_column(String(32), primary_key=True)
    cache_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value_json: Mapped[dict | list] = mapped_column(JSON)
    metrics_json: Mapped[dict | None] = mapped_column(JSON)
    content_hash: Mapped[str | None] = mapped_column(String(64), index=True)
    schema_version: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    last_accessed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    hit_count: Mapped[int] = mapped_column(Integer, default=0)
