"""Run 行数据库记录"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from sqlalchemy import and_, func, select, update
from sqlalchemy.sql.elements import ColumnElement

from deepresearcher.service.runs.transitions import assert_transition, transition_for
from deepresearcher.service.storage.tables import Run
from deepresearcher.service.storage.tables import utcnow as _utcnow

# 消息能到达就说明上一个持有者已停止续期，两个来源集合并即「可领取」。
_STARTABLE = transition_for("claim_resume").sources
_TAKEOVER = transition_for("adopt").sources


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


@dataclass(frozen=True)
class TerminalState:
    """run 落定终态时的状态，只装 done 帧要用的三项。"""

    status: str
    answer_mode: str
    report_available: bool


@dataclass(frozen=True)
class RunStateView:
    """消费判定所需的只读快照：任意状态与取消意图。租约到期与否由消息到达本身证明。"""

    status: str
    cancellation_requested_at: datetime | None


class RunStore:
    """按数据库行存储的 Run 领取、续期、释放与终态结算。"""

    def __init__(self, session_factory: Callable[[], Any]) -> None:
        self._session_factory = session_factory
        self._claim_lock = asyncio.Lock()

    async def acquire(self, run_id: str, *, worker_id: str) -> RunWork | None:
        """领取一次执行权，成功时写入 lease_owner 与 fence_token。

        参数：
            run_id: 消息体里的目标 run。
            worker_id: 领取者标识，写入 lease_owner。

        返回：
            RunWork | None: 领取成功 / 行已终态或带取消标记。
        """
        async with self._claim_lock:
            async with self._session_factory() as session:
                async with session.begin():
                    now = _utcnow()
                    previous = (
                        await session.execute(
                            select(Run.status).where(Run.id == run_id).with_for_update()
                        )
                    ).first()
                    if previous is None:
                        return None
                    row = (
                        await session.execute(
                            update(Run)
                            .where(
                                Run.id == run_id,
                                Run.cancellation_requested_at.is_(None),
                                Run.status.in_(_STARTABLE | _TAKEOVER),
                            )
                            .values(
                                status="running",
                                lease_owner=worker_id,
                                fence_token=Run.fence_token + 1,
                                started_at=func.coalesce(Run.started_at, now),
                            )
                            .returning(
                                Run.id,
                                Run.user_id,
                                Run.query,
                                Run.fence_token,
                                Run.resume_payload,
                            )
                        )
                    ).first()
        if row is None:
            return None
        return RunWork(
            run_id=row.id,
            user_id=row.user_id,
            query=row.query,
            # 来源已限死在 queued/interrupted/running，非 queued 就是「这行被领过」。
            resume=previous.status != "queued" or row.resume_payload is not None,
            resume_payload=row.resume_payload,
            lease_owner=worker_id,
            fence_token=row.fence_token,
        )

    async def inspect(self, run_id: str) -> RunStateView | None:
        """读消费判定所需的状态字段；行不存在返回 None。"""
        async with self._session_factory() as session:
            row = (
                await session.execute(
                    select(Run.status, Run.cancellation_requested_at).where(Run.id == run_id)
                )
            ).first()
        if row is None:
            return None
        return RunStateView(
            status=row.status,
            cancellation_requested_at=row.cancellation_requested_at,
        )

    async def verify_lease(self, work: RunWork) -> Literal["held", "cancelled", "lost"]:
        """判断租约和取消意图"""
        if not work.claimed:
            return "lost"
        async with self._session_factory() as session:
            row = (
                await session.execute(
                    select(Run.fence_token, Run.cancellation_requested_at).where(
                        Run.id == work.run_id
                    )
                )
            ).first()
        if row is None or row.fence_token != work.fence_token:
            return "lost"
        # 行离开 running 是执行自己写下的终态或挂起，不代表租约被夺走：收尾帧还要写库。
        # 真被夺走时 fence 会前进，上面已判失。
        return "cancelled" if row.cancellation_requested_at is not None else "held"

    async def release(
        self, work: RunWork, *, status: str, terminal_reason: str | None = None
    ) -> bool:
        """以 owner+fence_token CAS 释放领取，用于优雅 shutdown。

        参数：
            work: 本 worker 持有的领取凭据。
            status: 目标状态，须满足迁移表 running→status。
            terminal_reason: 可选终态原因。

        返回：
            bool: CAS 命中为 True；未领取或行已被接管为 False。
        """
        if not work.claimed:
            return False
        # WHERE 固定 running(所有权)，参数侧由迁移表把关。
        assert_transition("running", status)
        values: dict[str, Any] = {
            "status": status,
            "lease_owner": None,
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

    async def settle_cancelled(self, run_id: str) -> TerminalState | None:
        """把带取消标记的行置为 cancelled。

        参数：
            run_id: 目标 run。

        返回：
            写入后的状态与 done 帧所需字段。行已被别人置为终态时返回 None，不视为错误。
        """
        cancel = transition_for("cancel_pending")
        # running 也在结算范围内：消息能到达就说明原持有者已停止续期。
        abandoned = transition_for("cancel_running")
        return await self._write_terminal(
            and_(
                Run.id == run_id,
                Run.cancellation_requested_at.is_not(None),
                Run.status.in_(cancel.sources | abandoned.sources),
            ),
            status=cancel.target,
            reason=cancel.reason,
        )

    async def settle_exhausted(self, run_id: str) -> str | None:
        """投递次数耗尽的 run 置为终态，取消优先。"""
        cancel = transition_for("cancel_pending")
        if await self.settle_cancelled(run_id) is not None:
            return cancel.target
        exhaust = transition_for("settle_exhausted")
        settled = await self._write_terminal(
            # 取消若在两次写之间到达，这条谓词打空，返回 None 交给重投按取消结算，
            # 而不是把带取消标记的行判成 failed。
            and_(
                Run.id == run_id,
                Run.cancellation_requested_at.is_(None),
                Run.status.in_(exhaust.sources),
            ),
            status=exhaust.target,
            reason=exhaust.reason,
        )
        return exhaust.target if settled is not None else None

    async def reopen_exhausted(self, run_id: str) -> bool:
        """死信恢复入口：把重试耗尽的终态行退回 interrupted 等重投。"""
        reopen = transition_for("reopen_exhausted")
        exhausted_reason = transition_for("settle_exhausted").reason
        async with self._session_factory() as session:
            result = await session.execute(
                update(Run)
                .where(
                    Run.id == run_id,
                    Run.status.in_(reopen.sources),
                    Run.terminal_reason == exhausted_reason,
                )
                .values(
                    status=reopen.target,
                    terminal_reason=None,
                    error_message=None,
                    error_json=None,
                    review_json=None,
                    finished_at=None,
                    lease_owner=None,
                    fence_token=Run.fence_token + 1,
                )
            )
            await session.commit()
            return result.rowcount == 1

    async def _write_terminal(
        self, where: ColumnElement[bool], *, status: str, reason: str | None
    ) -> TerminalState | None:
        """按给定谓词把行写成终态；未命中返回 None。"""
        async with self._session_factory() as session:
            result = await session.execute(
                update(Run)
                .where(where)
                .values(
                    status=status,
                    terminal_reason=reason,
                    finished_at=_utcnow(),
                    lease_owner=None,
                    resume_payload=None,
                )
                .returning(Run.status, Run.answer_mode, Run.report_markdown)
            )
            row = result.one_or_none()
            await session.commit()
        if row is None:
            return None
        return TerminalState(
            status=row.status,
            answer_mode=row.answer_mode or "",
            report_available=bool(row.report_markdown),
        )
