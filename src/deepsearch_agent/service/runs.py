"""RunManager：单进程兼容期的 Run 受理、任务登记和恢复协调。

Graph 驱动、终态持久化和事件排水已下沉到 RunExecutor。后续迁移会再将
受理拆为 RunService，将进程内任务登记替换为持久 RunQueue/Scheduler。

三条贯穿本模块的不变式（都有对应回归测试）：

1. **run_id 由服务生成并强制注入**：行主键、ainvoke 输入、事件路由三者同一个
   id。引擎 instrumentation 的 setdefault 兜底只在 CLI 有意义；服务里缺了它，
   每个节点会各自生成新 id，事件与库行全部失联。
2. **终态恰好写一次**：Executor 在当前单循环内同步预占写入权；
   后续多 Worker 阶段改为 PostgreSQL lease + CAS。
3. **run_done 是流的唯一收尾符**：经同一 fanout.write 拿 seq，天然排在全部
   引擎事件之后；发布后才做最终 flush，所以断线重连从 DB 回放也能看到它。
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select, update

from deepsearch_agent.config import Settings
from deepsearch_agent.orchestration.graph import build_graph
from deepsearch_agent.service.event_store import RunEventStore
from deepsearch_agent.service.events import FanoutSink
from deepsearch_agent.service.executor import RunExecutor
from deepsearch_agent.service.models import Run, RunEvent
from deepsearch_agent.service.notifier import EventNotifier
from deepsearch_agent.service.queue import PostgresRunQueue, RunWork
from deepsearch_agent.service.run_service import QuotaExceededError as QuotaExceededError
from deepsearch_agent.service.run_service import RunService
from deepsearch_agent.service.settings import ServiceConfig
from deepsearch_agent.service.worker import EmbeddedWorker

TERMINAL_STATUSES = ("completed", "failed", "cancelled")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class RunManager:
    def __init__(
        self,
        *,
        settings: Settings,
        session_factory: Callable[[], Any],
        config: ServiceConfig,
        fanout: FanoutSink,
        http_client: Any,
        graph_factory: Callable[..., Any] = build_graph,
        checkpointer: Any = None,
        event_notifier: EventNotifier | None = None,
    ):
        self._checkpointer = checkpointer
        self._session_factory = session_factory
        self._config = config
        self._fanout = fanout
        self.event_notifier = event_notifier or EventNotifier()
        self.event_store = RunEventStore(
            session_factory,
            publish_persisted=fanout.publish_persisted,
            notifier=self.event_notifier,
        )
        self._run_service = RunService(session_factory=session_factory, config=config)
        self._queue = PostgresRunQueue(session_factory)
        self._executor = RunExecutor(
            settings=settings,
            session_factory=session_factory,
            config=config,
            fanout=fanout,
            event_store=self.event_store,
            http_client=http_client,
            graph_factory=graph_factory,
            checkpointer=checkpointer,
        )
        self._worker = EmbeddedWorker(
            queue=self._queue,
            executor=self._executor,
            max_running=config.max_global_running_runs,
            lease_seconds=config.worker_lease_seconds,
            heartbeat_seconds=config.worker_heartbeat_seconds,
            recover_expired=self._recover_expired,
        )
        # M2 兼容视图：权威运行事实已是 DB queued/running，测试与本地取消
        # 仍需观察当前 EmbeddedWorker 所持有的 asyncio.Task。
        self._tasks = self._worker.tasks

    # ---- 受理 ----

    async def start(self, user_id: int, query: str) -> str:
        query = query.strip()
        run_id = await self._run_service.create(user_id, query)
        self._fanout.open(run_id)
        await self._executor.publish_status(run_id, "queued")
        # queued 可能长时间等待，受理帧不能依赖 Executor 启动后的周期排水。
        await self._executor.flush_events(run_id)
        await self._worker.wake()
        return run_id

    async def cancel(self, user_id: int, run_id: str) -> Run:
        immediate = False
        async with self._session_factory() as session:
            run = await session.scalar(
                select(Run).where(Run.id == run_id, Run.user_id == user_id).with_for_update()
            )
            if run is None or run.user_id != user_id:
                raise LookupError(run_id)  # API 层统一转 404
            if run.status in TERMINAL_STATUSES:
                return run  # 幂等：已终结的运行原样返回
            run.cancellation_requested_at = _utcnow()
            if run.status in ("queued", "awaiting_input", "interrupted"):
                immediate = True
                run.status = "cancelled"
                run.terminal_reason = "user_cancelled"
                run.finished_at = _utcnow()
                run.resume_payload = None
                run.lease_owner = None
                run.lease_expires_at = None
            await session.commit()
        if immediate:
            await self._executor.publish_done(run_id)
            await self._executor.flush_events(run_id)
            self._fanout.close(run_id)
        else:
            self._worker.request_cancel(run_id)
        async with self._session_factory() as session:
            return await session.get(Run, run_id)

    # ---- 启动/停机 ----

    async def reconcile_startup(self) -> tuple[int, list[tuple[str, int, str]]]:
        """崩溃恢复分诊（步骤②）：活跃/中断残留行不再一律判死。

        - queued → 尚未获得执行槽，保留给 EmbeddedWorker 从头执行；
        - running/interrupted 且有 checkpoint → 可复活：原样留给 resume_runs（状态不改写，
          SSE 以 resuming 播报）；
        - 无 checkpoint 的 → 确认死亡：failed/server_restart + 补合成
          run_done 收尾帧（无此帧，历史详情页的 SSE 回放永远等不到结束，
          前端会无限重连）。
        checkpointer=None（CLI/SQLite 测试）时全部走判死路径，行为与既往一致。
        """
        resumable: list[tuple[str, int, str]] = []
        killed = 0
        await self._queue.reap_expired()
        async with self._session_factory() as session:
            stale = (
                await session.scalars(
                    select(Run).where(
                        (Run.status.in_(("queued", "interrupted")))
                        | ((Run.status == "running") & Run.lease_owner.is_(None))
                    )
                )
            ).all()
            for run in stale:
                if run.status == "queued":
                    max_seq = await session.scalar(
                        select(func.max(RunEvent.seq)).where(RunEvent.run_id == run.id)
                    )
                    self._fanout.open(run.id)
                    self._fanout.seed_seq(run.id, int(max_seq or 0))
                    continue
                if await self._has_checkpoint(run.id):
                    resumable.append((run.id, run.user_id, run.query))
                    continue
                killed += 1
                run.status = "failed"
                run.terminal_reason = "server_restart"
                run.error_message = "进程重启导致运行中断，请重新发起。"
                run.finished_at = _utcnow()
                max_seq = await session.scalar(
                    select(func.max(RunEvent.seq)).where(RunEvent.run_id == run.id)
                )
                done_record = {
                    "run_id": run.id,
                    "event_type": "run_done",
                    "seq": int(max_seq or 0) + 1,
                    "payload": {
                        "status": "failed",
                        "answer_mode": run.answer_mode or "",
                        "report_available": False,
                    },
                }
                session.add(
                    RunEvent(
                        run_id=run.id,
                        seq=done_record["seq"],
                        event_type="run_done",
                        record=done_record,
                    )
                )
                run.event_seq = done_record["seq"]
            await session.commit()
        return killed, resumable

    async def _has_checkpoint(self, run_id: str) -> bool:
        if self._checkpointer is None:
            return False
        tuple_ = await self._checkpointer.aget_tuple(
            {"configurable": {"thread_id": run_id, "checkpoint_ns": ""}}
        )
        return tuple_ is not None

    async def _recover_expired(self, work: RunWork) -> RunWork | None:
        """Classify an expired lease without assuming a checkpoint exists."""
        async with self._session_factory() as session:
            max_seq = await session.scalar(
                select(func.max(RunEvent.seq)).where(RunEvent.run_id == work.run_id)
            )
        self._fanout.open(work.run_id)
        self._fanout.seed_seq(work.run_id, int(max_seq or 0))
        if await self._has_checkpoint(work.run_id):
            await self._executor.publish_status(work.run_id, "resuming")
            await self._executor.flush_events(work.run_id)
            return RunWork(
                run_id=work.run_id,
                user_id=work.user_id,
                query=work.query,
                resume=True,
            )
        await self._executor.persist_status(
            work.run_id,
            status="failed",
            terminal_reason="lease_expired_without_checkpoint",
            error_message="运行中断且没有可恢复断点，请重新发起。",
        )
        await self._executor.publish_done(work.run_id)
        await self._executor.flush_events(work.run_id)
        self._fanout.close(work.run_id)
        return None

    async def resume_runs(self, pending: list[tuple[str, int, str]]) -> int:
        """复活有 checkpoint 的孤儿 run：astream(None) 从断点续跑。

        关键不变量：fanout 的 seq 计数器先从 run_events 的 max 续号——
        续跑事件的 seq 因此与上一世连续，主键不撞、前端去重不误杀。
        """
        for run_id, user_id, query in pending:
            async with self._session_factory() as session:
                max_seq = await session.scalar(
                    select(func.max(RunEvent.seq)).where(RunEvent.run_id == run_id)
                )
            self._fanout.open(run_id)
            self._fanout.seed_seq(run_id, int(max_seq or 0))
            await self._executor.publish_status(run_id, "resuming")
            await self._executor.flush_events(run_id)
            await self._worker.submit(
                RunWork(run_id=run_id, user_id=user_id, query=query, resume=True)
            )
        # 即使没有 checkpoint 恢复项，也要启动重启前已受理的 queued Run。
        await self._worker.wake()
        return len(pending)

    async def resume_with_input(self, user_id: int, run_id: str, answer: str) -> str:
        """向原 checkpoint 提交人类回答；同一暂停点只接受第一份。"""
        answer = answer.strip()
        if not answer:
            raise ValueError("澄清回答不能为空。")
        async with self._session_factory() as session:
            run = await session.get(Run, run_id)
            if run is None or run.user_id != user_id:
                raise LookupError(run_id)
            if run.status != "awaiting_input":
                raise RuntimeError(run.status)
        if not await self._has_checkpoint(run_id):
            raise RuntimeError("checkpoint_missing")
        async with self._session_factory() as session:
            result = await session.execute(
                update(Run)
                .where(
                    Run.id == run_id,
                    Run.user_id == user_id,
                    Run.status == "awaiting_input",
                    Run.resume_payload.is_(None),
                )
                .values(status="queued", resume_payload={"answer": answer})
            )
            await session.commit()
            if result.rowcount != 1:
                raise RuntimeError("already_resumed")
        self._fanout.open(run_id)
        await self._executor.publish_status(run_id, "queued")
        await self._executor.flush_events(run_id)
        await self._worker.wake()
        async with self._session_factory() as session:
            current = await session.get(Run, run_id)
        return current.status if current is not None else "queued"

    async def shutdown(self) -> None:
        await self._worker.shutdown()
