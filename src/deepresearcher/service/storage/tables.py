"""SQLAlchemy ORM 模型。users / runs / run_events。"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Run.status 用普通 str 而非枚举/CHECK 约束：增删状态不触发 ALTER。
# 终态是唯一需要跨层共识的子集——SSE 合成帧、claim 排除、租约结算共用这一份清单。
TERMINAL_STATUSES = ("completed", "failed", "cancelled")

_RETENTION_SWEEP_PREDICATE = text(
    "status IN ('"
    + "', '".join(TERMINAL_STATUSES)
    + "') AND finished_at IS NOT NULL AND retention_pruned_at IS NULL"
)

# 请求幂等键只在未终态范围内唯一：取消后带同一 request_id 重试时，旧行已是终态、
# 不占该键，新行可以正常受理。约束写进部分索引而不是列上，否则终态行会永久占键。
_ACTIVE_REQUEST_PREDICATE = text(
    "request_id IS NOT NULL AND status NOT IN ('" + "', '".join(TERMINAL_STATUSES) + "')"
)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class ProviderHealthRecord(Base):
    """搜索提供方账户级健康（跨 worker 共享的熔断位）。

    provider 主键；open_until 为该 key 判定不可用的恢复时刻（UTC）；
    reason 是稳定 user_code（invalid_key/insufficient_credit/forbidden/quota_exhausted）。
    """

    __tablename__ = "provider_health"

    provider: Mapped[str] = mapped_column(String(32), primary_key=True)
    reason: Mapped[str] = mapped_column(String(32))
    open_until: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    preferences: Mapped[dict | None] = mapped_column(JSON)

    runs: Mapped[list["Run"]] = relationship(back_populates="user")


class LoginThrottle(Base):
    """共享的登录尝试窗口；键只存 HMAC，不存原始 email/IP 值。"""

    __tablename__ = "login_throttles"

    key_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    window_started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    blocked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Run(Base):
    __tablename__ = "runs"
    __table_args__ = (
        # retention sweep 只需"终态、已到期、未清理"这一窄集。
        Index(
            "ix_runs_retention_sweep",
            "finished_at",
            postgresql_where=_RETENTION_SWEEP_PREDICATE,
        ),
        # 请求幂等的依据：同一用户未终态的行里 request_id 不重复，
        # 网络重试与双击提交都靠它查回既有 run。
        Index(
            "uq_runs_request_active",
            "user_id",
            "request_id",
            unique=True,
            postgresql_where=_ACTIVE_REQUEST_PREDICATE,
        ),
    )

    # 文本主键 "run-<hex>"：与引擎 run_id 同源，避免自增 id 泄露运行总量。
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    query: Mapped[str] = mapped_column(Text)
    # 客户端生成的幂等键；缺省为空表示调用方不要求去重（旧客户端）。
    request_id: Mapped[str | None] = mapped_column(String(64))
    # 两个标题：headline 由 Clarifier 浓缩问题(历史列表展示),
    # title 由 Writer 命名回答(报告 H1 与 PDF 文件名)。
    # create_all 只覆盖新库；既有库各补一次 ALTER TABLE runs ADD COLUMN。
    headline: Mapped[str | None] = mapped_column(String(80))
    status: Mapped[str] = mapped_column(String(16), default="queued")
    answer_mode: Mapped[str | None] = mapped_column(String(32))
    terminal_reason: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)
    report_markdown: Mapped[str | None] = mapped_column(Text)
    title: Mapped[str | None] = mapped_column(String(160))
    citations_json: Mapped[list | None] = mapped_column(JSON)
    # 创建时刻冻结的个人偏好全量快照：run 一生用它，用户中途改偏好不回溯；
    # resume 从本行重读同一份，不取用户现值。键为 AgentConfig 字段名。
    preferences_snapshot: Mapped[dict | None] = mapped_column(JSON)
    evidence_count: Mapped[int] = mapped_column(Integer, default=0)
    source_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # retention sweep 打标：终态且过期的 run 清完过程数据后置此，报告本体与行不删。
    retention_pruned_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # 当前执行者身份：只作写权谓词，不带时效。存活由 MQ 可见性证明，不看数据库。
    lease_owner: Mapped[str | None] = mapped_column(String(96), index=True)
    # 领取世代：fence token，每次领取或接管 +1，写权判定的权威字段。
    fence_token: Mapped[int] = mapped_column(Integer, default=0)
    cancellation_requested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # none_as_null=True：清空载荷必须写 SQL NULL，才能被 resume 的 is_(None) CAS 命中；
    resume_payload: Mapped[dict | None] = mapped_column(JSON(none_as_null=True))
    event_seq: Mapped[int] = mapped_column(Integer, default=0)
    llm_call_count: Mapped[int] = mapped_column(Integer, default=0)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cached_input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    external_request_count: Mapped[int] = mapped_column(Integer, default=0)
    peak_llm_concurrency: Mapped[int] = mapped_column(Integer, default=0)
    estimated_cost_cny: Mapped[float] = mapped_column(Numeric(14, 8), default=0)
    cache_hit_count: Mapped[int] = mapped_column(Integer, default=0)
    saved_external_request_count: Mapped[int] = mapped_column(Integer, default=0)

    user: Mapped["User"] = relationship(back_populates="runs")
    events: Mapped[list["RunEvent"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", passive_deletes=True
    )
    usage_records: Mapped[list["RunUsage"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", passive_deletes=True
    )


class RunEvent(Base):
    """研究过程事件的留档，供 SSE 断线重连回放与审计。"""

    __tablename__ = "run_events"
    __table_args__ = (UniqueConstraint("run_id", "dedup_key", name="uq_run_events_dedup"),)

    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), primary_key=True)
    seq: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_type: Mapped[str] = mapped_column(String(64))
    # create_all 只覆盖新库；既有库另补本列，并加上 UNIQUE (run_id, dedup_key) 约束
    # uq_run_events_dedup。
    dedup_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    record: Mapped[dict] = mapped_column(JSON)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    run: Mapped["Run"] = relationship(back_populates="events")


class RunUsage(Base):
    """一次计费或占用外部配额的操作。"""

    __tablename__ = "run_usage"
    __table_args__ = (Index("ix_run_usage_category_created_at", "category", "created_at"),)

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
    cost_cny: Mapped[float] = mapped_column(Numeric(14, 8), default=0)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    detail_json: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    run: Mapped["Run"] = relationship(back_populates="usage_records")
