"""Lease-based run worker shared by embedded and independent process modes."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable

from langgraph.types import Command

from deepresearcher.observability.logger import get_logger
from deepresearcher.observability.tracing.context import new_id
from deepresearcher.service.execution.executor import RunExecutor
from deepresearcher.service.runs.queue import (
    ClaimCapacitySaturated,
    PostgresRunQueue,
    RunWork,
)

logger = get_logger("deepresearcher.service.execution.worker")


class RunWorker:
    """Fill local slots on signals, with slow polling as a recovery path."""

    def __init__(
        self,
        *,
        queue: PostgresRunQueue,
        executor: RunExecutor,
        max_running: int,
        lease_seconds: int,
        heartbeat_seconds: int,
        poll_seconds: float = 15.0,
        worker_id: str | None = None,
        recover_expired: Callable[[RunWork], Awaitable[RunWork | None]] | None = None,
        after_reap: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._queue = queue
        self._executor = executor
        self._max_running = max(1, max_running)
        self._lease_seconds = max(10, lease_seconds)
        self._heartbeat_seconds = min(max(1, heartbeat_seconds), self._lease_seconds // 2)
        self._poll_seconds = max(0.05, poll_seconds)
        self._worker_id = worker_id or new_id("worker")
        self._recover_expired = recover_expired
        self._after_reap = after_reap
        # 瞬断容忍:连续 N 个心跳周期 renew 因 DB 错误失败才降级 lease_lost;
        # 心跳间隔 ≤ lease/2,预算耗尽时租约本已过期,reap 的判定与我们一致。
        self._heartbeat_failure_budget = 3
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._dispatch_tasks: set[asyncio.Task[None]] = set()
        self._claims: dict[str, RunWork] = {}
        self._explicit: deque[RunWork] = deque()
        self._explicit_ids: set[str] = set()
        self._dispatch_lock = asyncio.Lock()
        self._closed = False
        self._reaper_task: asyncio.Task[None] | None = None
        self._poll_task: asyncio.Task[None] | None = None

    @property
    def tasks(self) -> dict[str, asyncio.Task[None]]:
        """Compatibility view for cancellation and M1-era tests."""
        return self._tasks

    @property
    def worker_id(self) -> str:
        return self._worker_id

    async def start(self) -> None:
        """Start signal-ready dispatch and fallback polling; safe to call repeatedly."""
        if self._closed:
            return
        self._ensure_background_tasks()
        await self.wake()

    async def submit(self, work: RunWork) -> None:
        """Prioritize an already durable resume/recovery work item."""
        if work.run_id not in self._explicit_ids and work.run_id not in self._tasks:
            self._explicit.append(work)
            self._explicit_ids.add(work.run_id)
        await self.wake()

    def request_cancel(self, run_id: str) -> None:
        """Low-latency local hint; the database intent remains authoritative."""
        task = self._tasks.get(run_id)
        if task is not None and not task.done():
            self._executor.mark_cancellation_requested(run_id)
            task.cancel()

    async def cancel_if_requested(self, run_id: str) -> None:
        """Verify a notification against durable state before cancelling local work."""
        work = self._claims.get(run_id)
        if work is not None and await self._queue.cancellation_requested(work):
            self.request_cancel(run_id)

    async def wake(self) -> None:
        """Fill all currently free slots; safe to call after every state transition."""
        if self._closed:
            return
        self._ensure_background_tasks()
        async with self._dispatch_lock:
            while not self._closed and len(self._tasks) < self._max_running:
                preferred = self._take_explicit()
                work = await self._queue.claim(
                    worker_id=self._worker_id,
                    lease_seconds=self._lease_seconds,
                    preferred=preferred,
                )
                if isinstance(work, ClaimCapacitySaturated):
                    # 容量满 ≠ preferred 过期:弹出的显式任务放回队首、结束本轮,
                    # 等下一次信号/轮询/收割唤醒;静默丢弃会让 resume 行沉没。
                    if preferred is not None:
                        self._explicit.appendleft(preferred)
                        self._explicit_ids.add(preferred.run_id)
                    return
                if work is None:
                    # A stale explicit entry must not prevent ordinary queued work.
                    if preferred is not None:
                        continue
                    return
                task = asyncio.create_task(
                    self._execute_claimed(work),
                    name=f"research-worker-{work.run_id}",
                )
                self._tasks[work.run_id] = task
                self._claims[work.run_id] = work
                task.add_done_callback(lambda _task, run_id=work.run_id: self._on_task_done(run_id))

    def _ensure_background_tasks(self) -> None:
        if self._reaper_task is None:
            self._reaper_task = asyncio.create_task(
                self._reap_loop(), name=f"lease-reaper-{self._worker_id}"
            )
        if self._poll_task is None:
            self._poll_task = asyncio.create_task(
                self._poll_loop(), name=f"queue-poller-{self._worker_id}"
            )

    async def _poll_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._poll_seconds)
                try:
                    await self.wake()
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001 - 恢复路径自身不能再有单点:瞬断下轮重扫
                    logger.warning("queue_poll_db_error", exc_info=True)
        except asyncio.CancelledError:
            return

    async def _execute_claimed(self, work: RunWork) -> None:
        owner_task = asyncio.current_task()
        heartbeat = asyncio.create_task(
            self._heartbeat(work, owner_task), name=f"lease-heartbeat-{work.run_id}"
        )
        try:
            if work.resume:
                # claim 确实成功之后才广播恢复:被 cancel flag 排除或被他 worker
                # 抢走的 preferred 行,不该留下"恢复中"的幻影帧。
                await self._executor.publish_status(work.run_id, "resuming")
                await self._executor.flush_events(work.run_id)
            await self._executor.execute(
                work.run_id,
                work.user_id,
                work.query,
                resume=work.resume,
                resume_input=(
                    Command(resume=work.resume_payload)
                    if work.resume_payload is not None
                    else work.resume_input
                ),
                claim=work,
            )
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)

    async def _heartbeat(self, work: RunWork, owner_task: asyncio.Task[None] | None) -> None:
        """续租义务不允许静默退出:每轮只有三种结局——续上、确认取消/丢租、瞬断容忍。

        DB 抖动(renew 或取消检查抛错)只计失败数,预算耗尽才判 lease_lost;
        真被抢占/清空的 CAS 失败是确定信号,立即取消 owner。
        """
        transient_failures = 0
        try:
            while True:
                await asyncio.sleep(self._heartbeat_seconds)
                renewed = False
                cancelled: bool | None = None
                try:
                    renewed = await self._queue.renew(work, lease_seconds=self._lease_seconds)
                    if not renewed:
                        cancelled = await self._queue.cancellation_requested(work)
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001 - 基础设施瞬断不判健康执行的死刑
                    logger.warning("lease_heartbeat_db_error run_id=%s", work.run_id, exc_info=True)
                if renewed:
                    transient_failures = 0
                    continue
                if cancelled:
                    self._executor.mark_cancellation_requested(work.run_id)
                    if owner_task is not None:
                        owner_task.cancel()
                    return
                transient_failures += 1
                if cancelled is None and transient_failures < self._heartbeat_failure_budget:
                    # 无法区分"真丢租"与"DB 瞬断"时按后者处理:真丢租另有
                    # reap+settle 双保险,误杀却会让健康 run 白白重跑。
                    continue
                self._executor.mark_lease_lost(work.run_id)
                if owner_task is not None:
                    owner_task.cancel()
                return
        except asyncio.CancelledError:
            return

    async def _reap_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._heartbeat_seconds)
                try:
                    for expired in await self._queue.reap_expired():
                        recovered = (
                            await self._recover_expired(expired)
                            if self._recover_expired is not None
                            else None
                        )
                        if recovered is not None:
                            await self.submit(recovered)
                    if self._after_reap is not None:
                        await self._after_reap()
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001 - 收割是整个状态机最后的兜底,自己绝不能死
                    # 半途失败不补扫:reap_expired 每轮全量重扫,幂等吞掉瞬断即可。
                    logger.warning("lease_reap_db_error", exc_info=True)
        except asyncio.CancelledError:
            return

    def _take_explicit(self) -> RunWork | None:
        while self._explicit:
            work = self._explicit.popleft()
            self._explicit_ids.discard(work.run_id)
            if work.run_id not in self._tasks:
                return work
        return None

    def _on_task_done(self, run_id: str) -> None:
        self._tasks.pop(run_id, None)
        self._claims.pop(run_id, None)
        if not self._closed:
            task = asyncio.create_task(self.wake(), name="embedded-worker-dispatch")
            self._dispatch_tasks.add(task)
            task.add_done_callback(self._dispatch_tasks.discard)

    async def shutdown(self) -> None:
        self._closed = True
        background = [task for task in (self._poll_task, self._reaper_task) if task is not None]
        for task in background:
            task.cancel()
        for task in list(self._dispatch_tasks):
            task.cancel()
        await asyncio.gather(*background, *self._dispatch_tasks, return_exceptions=True)
        # 锁内快照:确保已穿过 _closed 检查、正等在 dispatch 锁上的 wake
        # 全部落地后,再决定 release 范围——否则会出现"release 跑完才被领取"
        # 的孤儿行(状态=running、进程已退出,只能等租约过期)。
        async with self._dispatch_lock:
            live = [
                (run_id, task, self._claims.get(run_id))
                for run_id, task in self._tasks.items()
                if not task.done()
            ]
        self._executor.mark_shutdown(run_id for run_id, _task, _work in live)
        for _run_id, task, _work in live:
            task.cancel()
        await asyncio.gather(*(task for _run_id, task, _work in live), return_exceptions=True)
        for _run_id, _task, work in live:
            if work is not None:
                await self._queue.release(
                    work, status="interrupted", terminal_reason="server_shutdown"
                )
