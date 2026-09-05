"""Lease-based run worker shared by embedded and independent process modes."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable

from langgraph.types import Command

from deepsearch_agent.observability.tracing.context import new_id
from deepsearch_agent.service.executor import RunExecutor
from deepsearch_agent.service.runs.queue import PostgresRunQueue, RunWork


class RunWorker:
    """Poll and fill process-local slots from the durable run queue."""

    def __init__(
        self,
        *,
        queue: PostgresRunQueue,
        executor: RunExecutor,
        max_running: int,
        lease_seconds: int,
        heartbeat_seconds: int,
        poll_seconds: float = 1.0,
        worker_id: str | None = None,
        recover_expired: Callable[[RunWork], Awaitable[RunWork | None]] | None = None,
    ) -> None:
        self._queue = queue
        self._executor = executor
        self._max_running = max(1, max_running)
        self._lease_seconds = max(10, lease_seconds)
        self._heartbeat_seconds = min(max(1, heartbeat_seconds), self._lease_seconds // 2)
        self._poll_seconds = max(0.05, poll_seconds)
        self._worker_id = worker_id or new_id("worker")
        self._recover_expired = recover_expired
        self._tasks: dict[str, asyncio.Task[None]] = {}
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
        """Start autonomous polling; safe to call more than once."""
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
                await self.wake()
        except asyncio.CancelledError:
            return

    async def _execute_claimed(self, work: RunWork) -> None:
        owner_task = asyncio.current_task()
        heartbeat = asyncio.create_task(
            self._heartbeat(work, owner_task), name=f"lease-heartbeat-{work.run_id}"
        )
        try:
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
        try:
            while True:
                await asyncio.sleep(self._heartbeat_seconds)
                if await self._queue.renew(work, lease_seconds=self._lease_seconds):
                    continue
                if await self._queue.cancellation_requested(work):
                    self._executor.mark_cancellation_requested(work.run_id)
                    if owner_task is not None:
                        owner_task.cancel()
                    return
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
                for expired in await self._queue.reap_expired():
                    recovered = (
                        await self._recover_expired(expired)
                        if self._recover_expired is not None
                        else None
                    )
                    if recovered is not None:
                        await self.submit(recovered)
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
            asyncio.create_task(self.wake(), name="embedded-worker-dispatch")

    async def shutdown(self) -> None:
        self._closed = True
        background = [
            task for task in (self._poll_task, self._reaper_task) if task is not None
        ]
        for task in background:
            task.cancel()
        await asyncio.gather(*background, return_exceptions=True)
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


# Compatibility import for callers/tests from the single-process migration stages.
EmbeddedWorker = RunWorker
