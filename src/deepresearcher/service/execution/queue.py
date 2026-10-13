"""持久的 PostgreSQL run 领取与租约管理。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import timedelta
from typing import Any

from sqlalchemy import func, select, update

from deepresearcher.service.persistence.models import Run
from deepresearcher.service.persistence.models import utcnow as _utcnow
from deepresearcher.service.runs.transitions import (
    apply_transition,
    assert_transition,
    transition_for,
)


@dataclass(frozen=True)
class RunWork:
    run_id: str
    user_id: int
    query: str
    resume: bool = False
    resume_payload: dict[str, Any] | None = None
    lease_owner: str | None = None
    fence_token: int = 0

    @property
    def claimed(self) -> bool:
        return self.lease_owner is not None and self.fence_token > 0


class PostgresRunQueue:
    """按数据库所有权与到期租约领取排队工作。

    单一胜者由条件 UPDATE 本身保证（status ∈ sources + 取消守卫 +
    fence_token 递增）：对行来说是原子 CAS，落败的领取者看到 rowcount 0。
    PostgreSQL 用 ``FOR UPDATE SKIP LOCKED`` 选候选，多个领取者因此
    互不等待。``_claim_lock`` 只串行化本实例自身的 poll/wake 路径——
    多宿主正确性（如测试中两个 worker 共用一个 SQLite 文件）依赖 CAS，
    不依赖该锁。

    本类**不**约束跨实例的同时执行总量：并发上限由调用方的本地槽位数
    （``RunWorker.max_running``）决定。多副本部署时全局总量等于
    槽位数乘以副本数，这是预期行为——上限的语义是「当前资源能支撑的
    并发数」，资源扩容后它本就该跟着涨。
    """

    def __init__(self, session_factory: Callable[[], Any]) -> None:
        self._session_factory = session_factory
        self._claim_lock = asyncio.Lock()

    async def claim(
        self,
        *,
        worker_id: str,
        lease_seconds: int,
        preferred: RunWork | None = None,
    ) -> RunWork | None:
        """领取一行排队工作并建立租约。

        参数:
            worker_id: 领取者标识，写入 lease_owner。
            lease_seconds: 租约时长。
            preferred: 显式待领取项（resume/恢复路径）。

        返回:
            RunWork | None: 领取成功 / 本轮无可领取行
            （含 preferred 已被取消或已被其他 worker 领走）。
        """
        async with self._claim_lock:
            async with self._session_factory() as session:
                now = _utcnow()
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
                        fence_token=Run.fence_token + 1,
                        started_at=func.coalesce(Run.started_at, now),
                    )
                    .returning(Run.id, Run.user_id, Run.query, Run.fence_token, Run.resume_payload)
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
                    fence_token=row.fence_token,
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
                    Run.fence_token == work.fence_token,
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
                    Run.fence_token == work.fence_token,
                )
            )
            return value is not None

    async def release(
        self, work: RunWork, *, status: str, terminal_reason: str | None = None
    ) -> bool:
        """以 owner+fence_token CAS 释放领取，通常用于优雅 shutdown。

        参数:
            work: 本 worker 持有的领取凭据。
            status: 目标状态，须满足迁移表 running→status。
            terminal_reason: 可选终态原因。

        返回:
            bool: CAS 命中为 True；未领取或行已被接管为 False。

        抛出:
            IllegalTransitionError: status 不是 running 的合法目标时。
        """
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
                    Run.fence_token == work.fence_token,
                )
                .values(**values)
            )
            await session.commit()
            return result.rowcount == 1

    async def settle_cancellations(self) -> list[RunWork]:
        """取消意图是持久事实，其执行不依赖 executor 存活。

        带取消标记的行里，仍在 running 的由 heartbeat 与 executor 自写终态；
        只有 executor 退出时未写终态的 interrupted 行需要这里补写。claim 的
        WHERE 始终排除带标记的行，不做 settle 这些行就会永久停留。

        返回:
            被终态化的行，仅含身份字段，供调用方补投递。
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
        """把租约过期的 running 领取退回 ``interrupted``，供 checkpoint 续跑。

        返回:
            list[RunWork]: 被退回的行（仅身份字段，无领取凭据）。
        """
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
