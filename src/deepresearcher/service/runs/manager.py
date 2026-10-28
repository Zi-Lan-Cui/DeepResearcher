"""持久 Run 命令的 API 控制层。manager 受理、取消、恢复 run。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sqlalchemy import select, update

from deepresearcher.observability.events.names import EventName
from deepresearcher.service.coordination.signals import PostgresSignalBus
from deepresearcher.service.events.buffer import RunEventBuffer
from deepresearcher.service.messaging.publisher import DispatchPublisher
from deepresearcher.service.runs.admission import QuotaExceededError as QuotaExceededError
from deepresearcher.service.runs.admission import RunAdmission
from deepresearcher.service.runs.store import TerminalState
from deepresearcher.service.runs.transitions import apply_transition, transition_for
from deepresearcher.service.settings import ServiceConfig
from deepresearcher.service.storage.tables import TERMINAL_STATUSES, Run, RunEvent
from deepresearcher.service.storage.tables import utcnow as _utcnow


class RunManager:
    """持久化用户命令，并向 HTTP 层暴露持久事件。"""

    def __init__(
        self,
        *,
        session_factory: Callable[[], Any],
        config: ServiceConfig,
        buffer: RunEventBuffer,
        publisher: DispatchPublisher,
        checkpointer: Any = None,
        signal_bus: PostgresSignalBus | None = None,
    ) -> None:
        self._checkpointer = checkpointer
        self._session_factory = session_factory
        self._config = config
        self._buffer = buffer
        self._publisher = publisher
        self.signal_bus = signal_bus or PostgresSignalBus()
        # 只读句柄：cancel 即时分支要 tail 判终态；写库与 notify 归 Buffer 统管。
        self.event_store = buffer.store
        self._admission = RunAdmission(
            session_factory=session_factory, config=config, publisher=publisher
        )

    async def recent_runs(self, user_id: int, limit: int = 50) -> list[Run]:
        """用户历史列表(倒序):web 层读取 Run 的唯一路径。"""
        async with self._session_factory() as session:
            return list(
                (
                    await session.scalars(
                        select(Run)
                        .where(Run.user_id == user_id)
                        .order_by(Run.created_at.desc())
                        .limit(limit)
                    )
                ).all()
            )

    async def pending_clarification_payload(self, run: Run) -> dict[str, Any] | None:
        """run 等待输入时返回最近一条 clarification_requested 的 payload；其余状态为 None。

        事件词汇与查询结构集中在此；调用方只把它交给 project_clarification。
        """
        if run.status != "awaiting_input":
            return None
        async with self._session_factory() as session:
            row = await session.scalar(
                select(RunEvent)
                .where(
                    RunEvent.run_id == run.id,
                    RunEvent.event_type == EventName.CLARIFICATION_REQUESTED,
                )
                .order_by(RunEvent.seq.desc())
                .limit(1)
            )
        if row is None:
            return None
        payload = row.record.get("payload", {})
        return payload if isinstance(payload, dict) else None

    async def start(
        self,
        user_id: int,
        query: str,
        preferences_snapshot: dict[str, int] | None = None,
        *,
        request_id: str | None = None,
    ) -> str:
        """受理新研究问题：创建 queued 行并发一条分派消息。

        preferences_snapshot 为创建时刻冻结的个人偏好全量值，随行持久化；
        缺省为 NULL，执行期回服务器默认。

        request_id 是客户端幂等键：同一键在未终态范围内只受理一次，重复提交
        返回既有 run 的 id，也不重复分派。

        返回：
            run 的 id。
        """
        run_id = await self._admission.create(
            user_id, query.strip(), preferences_snapshot, request_id=request_id
        )
        self._buffer.open(run_id)
        self._buffer.write_status(run_id, "queued")
        await self._buffer.flush(run_id)
        return run_id

    async def cancel(self, user_id: int, run_id: str) -> Run:
        """请求取消一个 run。"""
        done: TerminalState | None = None
        async with self._session_factory() as session:
            run = await session.scalar(
                select(Run).where(Run.id == run_id, Run.user_id == user_id).with_for_update()
            )
            if run is None or run.user_id != user_id:
                raise LookupError(run_id)
            if run.status in TERMINAL_STATUSES:
                return run
            run.cancellation_requested_at = _utcnow()
            if run.status in transition_for("cancel_pending").sources:
                apply_transition(run, "cancel_pending", now=_utcnow())
                # commit 会让属性过期，之后读 run.status 会隐式再查一次库；
                # done 帧要的三个值在这里取走。
                done = TerminalState(
                    status=run.status,
                    answer_mode=run.answer_mode or "",
                    report_available=bool(run.report_markdown),
                )
            await session.commit()
        if done is not None:
            self._buffer.open(run_id)
            self._buffer.write_done(
                run_id,
                status=done.status,
                answer_mode=done.answer_mode,
                report_available=done.report_available,
            )
            await self._buffer.flush(run_id)
            self._buffer.close(run_id)
        else:
            await self.signal_bus.notify_cancel(run_id)
        async with self._session_factory() as session:
            return await session.get(Run, run_id)

    async def terminal_state(self, run_id: str) -> dict[str, Any] | None:
        """SSE 合成终止帧用：run 已终态时返回 done 帧载荷要素，否则返回 None。"""
        async with self._session_factory() as session:
            run = await session.get(Run, run_id)
        if run is None or run.status not in TERMINAL_STATUSES:
            return None
        return {
            "status": run.status,
            "answer_mode": run.answer_mode or "",
            "report_available": bool(run.report_markdown),
        }

    async def _has_checkpoint(self, run_id: str) -> bool:
        if self._checkpointer is None:
            return False
        tuple_ = await self._checkpointer.aget_tuple(
            {"configurable": {"thread_id": run_id, "checkpoint_ns": ""}}
        )
        return tuple_ is not None

    async def resume_with_input(self, user_id: int, run_id: str, answer: str) -> str:
        """提交澄清回答并把 run 重新入队。"""
        answer = answer.strip()
        if not answer:
            raise ValueError("澄清回答不能为空。")
        resume_transition = transition_for("answer_resume")
        async with self._session_factory() as session:
            run = await session.get(Run, run_id)
            if run is None or run.user_id != user_id:
                raise LookupError(run_id)
            if run.status not in resume_transition.sources:
                raise RuntimeError(run.status)
        if not await self._has_checkpoint(run_id):
            raise RuntimeError("checkpoint_missing")

        async def local_transaction() -> None:
            """把回答写进 resume_payload 并把状态改回 queued。"""
            async with self._session_factory() as session:
                result = await session.execute(
                    update(Run)
                    .where(
                        Run.id == run_id,
                        Run.user_id == user_id,
                        Run.status.in_(resume_transition.sources),
                        Run.resume_payload.is_(None),
                    )
                    .values(status=resume_transition.target, resume_payload={"answer": answer})
                )
                await session.commit()
                if result.rowcount != 1:
                    # 已被取消或已被别人回答：确定拒绝，消息随回滚一起丢弃。
                    raise RuntimeError("already_resumed")

        await self._publisher.dispatch(run_id, local_transaction)
        self._buffer.open(run_id)
        self._buffer.write_status(run_id, "queued")
        await self._buffer.flush(run_id)
        return resume_transition.target
