"""Durable PostgreSQL run claiming and lease management."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import timedelta
from typing import Any

from sqlalchemy import func, select, text, update

from deepresearcher.service.coordination import RUN_CLAIM_CAPACITY_LOCK_ID
from deepresearcher.service.persistence.models import Run
from deepresearcher.service.persistence.models import utcnow as _utcnow
from deepresearcher.service.runs.transitions import (
    apply_transition,
    assert_transition,
    transition_for,
)


@dataclass(frozen=True)
class ClaimCapacitySaturated:
    """全局并发上限已满、本轮无法领取——与"没有可领取的行"是两种语义。

    调用方(worker 的 wake)据此把显式 preferred 放回队列;
    若混同为 None,容量满瞬间从 recover/settle 弹出的 resume 任务会被吞掉,
    该 interrupted 行要停留到下次进程重启。
    """


CAPACITY_SATURATED = ClaimCapacitySaturated()


@dataclass(frozen=True)
class RunWork:
    run_id: str
    user_id: int
    query: str
    resume: bool = False
    resume_input: Any = None
    resume_payload: dict[str, Any] | None = None
    lease_owner: str | None = None
    attempt: int = 0

    @property
    def claimed(self) -> bool:
        return self.lease_owner is not None and self.attempt > 0


class PostgresRunQueue:
    """Claim queued work with database ownership and expiring leases.

    Single-winner comes from the conditional UPDATE itself (status ∈ sources +
    cancellation guard + attempt bump): it is an atomic CAS at the row, so a
    losing claimer sees rowcount 0. PostgreSQL adds ``FOR UPDATE SKIP LOCKED``
    candidate selection and an xact advisory lock around the count+claim
    capacity check. ``_claim_lock`` only serializes this instance's own
    poll/wake paths — multi-host correctness (e.g. two workers on one SQLite
    file in tests) rests on the CAS, not on that lock.
    """

    def __init__(
        self, session_factory: Callable[[], Any], *, max_global_running: int | None = None
    ) -> None:
        self._session_factory = session_factory
        self._max_global_running = (
            max(1, max_global_running) if max_global_running is not None else None
        )
        self._claim_lock = asyncio.Lock()

    async def claim(
        self,
        *,
        worker_id: str,
        lease_seconds: int,
        preferred: RunWork | None = None,
    ) -> RunWork | ClaimCapacitySaturated | None:
        async with self._claim_lock:
            async with self._session_factory() as session:
                now = _utcnow()
                if self._max_global_running is not None:
                    # PostgreSQL 上用事务 advisory lock 串行化“计数+领取”；
                    # SQLite 测试/单进程兼容路径由 _claim_lock 保护。
                    bind = session.get_bind()
                    if bind.dialect.name == "postgresql":
                        await session.execute(
                            text("SELECT pg_advisory_xact_lock(:lock_id)"),
                            {"lock_id": RUN_CLAIM_CAPACITY_LOCK_ID},
                        )
                    running = await session.scalar(
                        select(func.count()).select_from(Run).where(Run.status == "running")
                    )
                    if int(running or 0) >= self._max_global_running:
                        await session.rollback()
                        return CAPACITY_SATURATED
                claim_name = "claim"
                if preferred is None:
                    candidate = (
                        select(Run.id)
                        .where(Run.status == "queued", Run.cancellation_requested_at.is_(None))
                        .order_by(Run.created_at, Run.id)
                        .limit(1)
                        .with_for_update(skip_locked=True)
                        .scalar_subquery()
                    )
                else:
                    claim_name = "claim_resume" if preferred.resume else "claim"
                    candidate = preferred.run_id
                claim_transition = transition_for(claim_name)
                result = await session.execute(
                    update(Run)
                    .where(
                        Run.id == candidate,
                        Run.status.in_(claim_transition.sources),
                        Run.cancellation_requested_at.is_(None),
                    )
                    .values(
                        status=claim_transition.target,
                        lease_owner=worker_id,
                        lease_expires_at=now + timedelta(seconds=lease_seconds),
                        attempt=Run.attempt + 1,
                        started_at=func.coalesce(Run.started_at, now),
                    )
                    .returning(Run.id, Run.user_id, Run.query, Run.attempt, Run.resume_payload)
                )
                row = result.first()
                await session.commit()
                if row is None:
                    return None
                base = preferred or RunWork(run_id=row.id, user_id=row.user_id, query=row.query)
                return replace(
                    base,
                    resume=base.resume or row.resume_payload is not None,
                    resume_payload=row.resume_payload,
                    lease_owner=worker_id,
                    attempt=row.attempt,
                )

    async def renew(self, work: RunWork, *, lease_seconds: int) -> bool:
        if not work.claimed:
            return False
        async with self._session_factory() as session:
            result = await session.execute(
                update(Run)
                .where(
                    Run.id == work.run_id,
                    Run.status == "running",
                    Run.lease_owner == work.lease_owner,
                    Run.attempt == work.attempt,
                    Run.cancellation_requested_at.is_(None),
                )
                .values(lease_expires_at=_utcnow() + timedelta(seconds=lease_seconds))
            )
            await session.commit()
            return result.rowcount == 1

    async def cancellation_requested(self, work: RunWork) -> bool:
        if not work.claimed:
            return False
        async with self._session_factory() as session:
            value = await session.scalar(
                select(Run.cancellation_requested_at).where(
                    Run.id == work.run_id,
                    Run.status == "running",
                    Run.lease_owner == work.lease_owner,
                    Run.attempt == work.attempt,
                )
            )
            return value is not None

    async def release(
        self, work: RunWork, *, status: str, terminal_reason: str | None = None
    ) -> bool:
        """Release a claim with owner+attempt CAS, normally for graceful shutdown."""
        if not work.claimed:
            return False
        # WHERE 固定 running(所有权),参数侧由迁移表把关。
        assert_transition("running", status)
        values: dict[str, Any] = {
            "status": status,
            "lease_owner": None,
            "lease_expires_at": None,
            "finished_at": None,
        }
        if terminal_reason is not None:
            values["terminal_reason"] = terminal_reason
        async with self._session_factory() as session:
            result = await session.execute(
                update(Run)
                .where(
                    Run.id == work.run_id,
                    Run.status == "running",
                    Run.lease_owner == work.lease_owner,
                    Run.attempt == work.attempt,
                )
                .values(**values)
            )
            await session.commit()
            return result.rowcount == 1

    async def settle_cancellations(self) -> list[RunWork]:
        """取消意图是持久事实,落地不依赖 executor 活着。

        带 flag 的行里,running 归 heartbeat→executor 自写终态;只有
        executor 退出时未写终态的 interrupted 行需要这里补写。claim 的
        WHERE 永远排除带 flag 的行,不 settle 它们就永久停留。
        """
        async with self._session_factory() as session:
            settle_transition = transition_for("settle_cancelled")
            rows = (
                await session.scalars(
                    select(Run)
                    .where(
                        Run.cancellation_requested_at.is_not(None),
                        Run.status.in_(settle_transition.sources),
                    )
                    .with_for_update(skip_locked=True)
                )
            ).all()
            work = [RunWork(run_id=run.id, user_id=run.user_id, query=run.query) for run in rows]
            for run in rows:
                apply_transition(run, "settle_cancelled", now=_utcnow())
            await session.commit()
            return work

    async def reap_expired(self) -> list[RunWork]:
        """Return expired running claims to ``interrupted`` for checkpoint resume."""
        async with self._claim_lock:
            async with self._session_factory() as session:
                reap_transition = transition_for("reap")
                rows = (
                    await session.scalars(
                        select(Run)
                        .where(
                            Run.status.in_(reap_transition.sources),
                            Run.lease_expires_at.is_not(None),
                            Run.lease_expires_at < _utcnow(),
                        )
                        .with_for_update(skip_locked=True)
                    )
                ).all()
                work = [
                    RunWork(run_id=run.id, user_id=run.user_id, query=run.query) for run in rows
                ]
                for run in rows:
                    apply_transition(run, "reap", now=_utcnow())
                await session.commit()
                return work
