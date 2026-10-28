"""为独立执行进程提供 run worker。"""

from __future__ import annotations

import asyncio

from langgraph.types import Command
from sqlalchemy.exc import OperationalError

from deepresearcher.observability.logging_config import get_logger
from deepresearcher.observability.span import new_id
from deepresearcher.service.events.buffer import RunEventBuffer
from deepresearcher.service.execution.executor import RunExecutor
from deepresearcher.service.messaging.receiver import DispatchDelivery, DispatchReceiver
from deepresearcher.service.runs.store import RunStateView, RunStore, RunWork
from deepresearcher.service.storage.tables import TERMINAL_STATUSES

logger = get_logger("deepresearcher.service.execution.consumer")

#: 拉取失败后的退避：Broker 抖动时循环里没有别的挂起点。
_RECEIVE_RETRY_BACKOFF_SECONDS = 5.0

#: 空轮询后的短退避：Broker 立刻返回空批次时，循环里没有别的挂起点。
_EMPTY_POLL_BACKOFF_SECONDS = 0.05


class RunConsumer:
    """按分派消息填充本地槽位；消息是该 run 唯一的执行来源。"""

    def __init__(
        self,
        *,
        store: RunStore,
        executor: RunExecutor,
        buffer: RunEventBuffer,
        receiver: DispatchReceiver,
        max_running: int,
        lease_check_seconds: int,
        invisible_seconds: int,
        visibility_renew_seconds: int,
        worker_id: str | None = None,
        max_attempts: int = 15,
    ) -> None:
        self._store = store
        self._executor = executor
        self._buffer = buffer
        self._receiver = receiver
        self._max_running = max_running
        self._visibility_renew_seconds = max(1, visibility_renew_seconds)
        # 确认要比续租更勤：续租只在确认通过时做，拍子太慢会把一次读库抖动拖成租约过期。
        self._lease_check_seconds = min(
            max(1, lease_check_seconds), self._visibility_renew_seconds // 2
        )
        self._invisible_seconds = max(invisible_seconds, self._visibility_renew_seconds * 2)
        self._worker_id = worker_id or new_id("worker")
        self._max_attempts = max(2, max_attempts)
        # 瞬断容忍：连续这么多个确认周期都探不到租约才停手。取值对齐「可见性窗口」，
        # 续不上的时间超过它，别处就已经可以把这行接走。
        self._lease_check_failure_budget = max(
            2, self._invisible_seconds // max(1, self._lease_check_seconds)
        )
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._claims: dict[str, RunWork] = {}
        self._receipts: dict[str, DispatchDelivery] = {}
        # 槽位在拉取之前获取：满了就连消息都不取，不把消息囤在内存里等。
        self._slots = asyncio.Semaphore(max_running)
        self._closed = False
        self._receive_task: asyncio.Task[None] | None = None

    @property
    def tasks(self) -> dict[str, asyncio.Task[None]]:
        """在执行的本地任务视图。"""
        return self._tasks

    @property
    def worker_id(self) -> str:
        return self._worker_id

    def cancel_local(self, run_id: str) -> None:
        """只停本地这一份执行，不写数据库；取消的权威在库里。"""
        task = self._tasks.get(run_id)
        if task is not None and not task.done():
            task.cancel()

    async def cancel_if_confirmed(self, run_id: str) -> None:
        """通知只是线索，回查数据库确认有取消意图后才停本地执行。"""
        work = self._claims.get(run_id)
        if work is not None and await self._store.verify_lease(work) == "cancelled":
            self.cancel_local(run_id)

    async def _ack_quietly(self, delivery: DispatchDelivery) -> None:
        try:
            await self._receiver.ack(delivery)
        except Exception:  # noqa: BLE001 - 确认失败只会导致一次多余重投，无需上抛
            logger.warning("dispatch_ack_failed run_id=%s", delivery.message.run_id, exc_info=True)

    async def start(self) -> None:
        """启动消息拉取；可安全重复调用。"""
        if self._closed or self._receive_task is not None:
            return
        self._receive_task = asyncio.create_task(
            self._receive_loop(), name=f"dispatch-receiver-{self._worker_id}"
        )

    async def _receive_loop(self) -> None:
        """槽位可用才拉取；拉到的每条消息都必须有明确结局：确认，或留待重投。"""
        try:
            while not self._closed:
                await self._slots.acquire()
                if self._closed:
                    self._slots.release()
                    return
                try:
                    delivery = await self._receiver.receive()
                except asyncio.CancelledError:
                    self._slots.release()
                    raise
                except Exception:  # noqa: BLE001 - 拉取失败退化为下一轮重试
                    self._slots.release()
                    logger.warning("dispatch_receive_failed", exc_info=True)
                    await asyncio.sleep(_RECEIVE_RETRY_BACKOFF_SECONDS)
                    continue
                if delivery is None:
                    self._slots.release()
                    # 空批次必须让出控制权：长轮询正常情况下会阻塞，但 Broker 快速
                    # 返回空列表时，这个循环里没有别的挂起点。
                    await asyncio.sleep(_EMPTY_POLL_BACKOFF_SECONDS)
                    continue
                await self._accept(delivery)
        except asyncio.CancelledError:
            return

    async def _accept(self, delivery: DispatchDelivery) -> None:
        """按数据库当前行给这条消息定性，必要时建任务并持有槽位。"""
        run_id = delivery.message.run_id
        try:
            state = await self._store.inspect(run_id)
        except Exception:  # noqa: BLE001 - 读不到状态就不确认，等重投
            self._slots.release()
            logger.warning("dispatch_state_read_failed run_id=%s", run_id, exc_info=True)
            return
        if state is None:
            # 行不存在：消息属于一个已被清理的尝试，确认后丢弃。
            await self._ack_quietly(delivery)
            self._slots.release()
            return
        if state.status in TERMINAL_STATUSES:
            await self._ack_quietly(delivery)
            self._slots.release()
            return
        if state.cancellation_requested_at is not None:
            # 取消优先于一切：消息能到达就说明原持有者已停止续期，直接替它结算。
            # 结算不动时不确认——那说明写库失败，留给下一次投递重判。
            settled = await self._settle_cancelled(run_id)
            if settled:
                await self._ack_quietly(delivery)
            self._slots.release()
            return
        if state.status == "awaiting_input":
            # 等待澄清时原消息已经确认，用户回答会另发一条分派消息。
            # 这条不确认就会一直重投，直到投递次数耗尽被当作死信。
            await self._ack_quietly(delivery)
            self._slots.release()
            return
        if delivery.delivery_attempt >= self._max_attempts:
            # 投递次数耗尽：数据库仍说可跑，却一直跑不起来。
            # 这个判定放在读库成功之后——读不到库时的重投增长是故障本身，不是毒消息，
            # 那种情况走上面的异常分支，不确认、继续重投。
            settled = await self._exhaust(delivery, state)
            self._slots.release()
            if settled:
                await self._ack_quietly(delivery)
            return
        try:
            work = await self._store.acquire(run_id, worker_id=self._worker_id)
        except Exception:  # noqa: BLE001 - CAS 失败按「没抢到」处理，不确认
            self._slots.release()
            logger.warning("dispatch_acquire_failed run_id=%s", run_id, exc_info=True)
            return
        if work is None:
            # 只剩一种成因：inspect 与 acquire 之间行状态被改（终态化或被取消）。
            # 不确认，等下一次投递重走判定即可定性。
            self._slots.release()
            logger.warning("dispatch_acquire_raced run_id=%s", run_id)
            return
        task = asyncio.create_task(
            self._execute_claimed(work, delivery),
            name=f"research-worker-{run_id}",
        )
        self._tasks[run_id] = task
        self._claims[run_id] = work
        self._receipts[run_id] = delivery
        task.add_done_callback(lambda _task, rid=run_id: self._on_task_done(rid))

    async def _execute_claimed(self, work: RunWork, delivery: DispatchDelivery) -> None:
        owner_task = asyncio.current_task()
        lease_task = asyncio.create_task(
            self._hold_lease(work, owner_task, delivery),
            name=f"lease-hold-{work.run_id}",
        )
        try:
            if work.resume:
                # acquire 成功后才广播恢复：被取消标记排除或被他 worker 抢走的行，
                # 不应发出误报的 resuming 帧。
                self._buffer.open(work.run_id)
                self._buffer.write_status(work.run_id, "resuming")
                await self._buffer.flush(work.run_id)
            await self._executor.execute(
                work,
                resume_input=(
                    Command(resume=work.resume_payload) if work.resume_payload is not None else None
                ),
            )
        finally:
            lease_task.cancel()
            await asyncio.gather(lease_task, return_exceptions=True)
        # 执行返回即终态已经提交（executor 把状态与终态事件放在同一事务里），此刻才确认。
        # 早一步确认会让「终态已提交、消息却丢了」变成不可恢复。
        await self._ack_quietly(delivery)

    async def _hold_lease(
        self,
        work: RunWork,
        owner_task: asyncio.Task[None] | None,
        delivery: DispatchDelivery,
    ) -> None:
        """每拍确认租约仍归本 worker，并带回用户的取消意图。

        确认连续通过到续期间隔，才把消息的不可见时长续一次。首拍不等待：执行起点
        就该确认一次租约，否则接管与取消要多等一个完整确认周期。
        """
        transient_failures = 0
        visibility_tick = 0
        try:
            while True:
                verdict: str | None = None
                try:
                    verdict = await self._store.verify_lease(work)
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001 - 瞬断不应把健康执行判为失败
                    logger.warning("lease_check_db_error run_id=%s", work.run_id, exc_info=True)
                if verdict == "held":
                    transient_failures = 0
                    visibility_tick += self._lease_check_seconds
                    if visibility_tick >= self._visibility_renew_seconds:
                        visibility_tick = 0
                        try:
                            await self._receiver.extend_visibility(
                                delivery, self._invisible_seconds
                            )
                        except Exception:  # noqa: BLE001 - 续不上可见性只会多一次重投
                            logger.warning(
                                "lease_visibility_renew_failed run_id=%s",
                                work.run_id,
                                exc_info=True,
                            )
                elif verdict == "cancelled":
                    if owner_task is not None:
                        owner_task.cancel()
                    return
                else:
                    transient_failures += 1
                    # 真丢权（lost）立即停手；探不清（DB 瞬断）留足预算——误判会让健康
                    # run 白白重试，而真丢权还有可见性到期与 fence 接着收。
                    if (
                        verdict is not None
                        or transient_failures >= self._lease_check_failure_budget
                    ):
                        self._executor.mark_lease_lost(work.run_id)
                        if owner_task is not None:
                            owner_task.cancel()
                        return
                await asyncio.sleep(self._lease_check_seconds)
        except asyncio.CancelledError:
            return

    async def _settle_cancelled(self, run_id: str) -> bool:
        """把取消意图落成终态并补 done 帧；返回是否真的写了终态。"""
        try:
            written = await self._store.settle_cancelled(run_id)
        except Exception:  # noqa: BLE001 - 结算失败留给重投，不让消息卡住槽位
            logger.warning("dispatch_cancel_settle_failed run_id=%s", run_id, exc_info=True)
            return False
        if written is None:
            return False
        self._buffer.open(run_id)
        self._buffer.write_done(
            run_id,
            status=written.status,
            answer_mode=written.answer_mode,
            report_available=written.report_available,
        )
        await self._buffer.flush(run_id)
        self._buffer.close(run_id)
        return True

    async def _exhaust(self, delivery: DispatchDelivery, state: RunStateView) -> bool:
        """投递耗尽：把行结案，返回是否需要确认这条消息。

        结案时取消优先：带取消标记走 cancelled，否则 failed 且 reason 为
        dispatch_attempts_exhausted。

        参数：
            delivery: 耗尽的投递。
            state: 读到的行快照，仅用于日志。

        返回：
            bool: 行已落终态为 True。写不进数据库时为 False，调用方不得确认——
            消息是这一行唯一的路标。
        """
        run_id = delivery.message.run_id
        try:
            settled = await self._store.settle_exhausted(run_id)
        except Exception:  # noqa: BLE001 - 写不进去就不确认：留给重投与下一轮判定
            logger.warning("dispatch_exhaust_settle_failed run_id=%s", run_id, exc_info=True)
            return False
        logger.error(
            "dispatch_attempts_exhausted run_id=%s attempts=%d status=%s settled=%s",
            run_id,
            delivery.delivery_attempt,
            state.status,
            settled,
        )
        return settled is not None

    def _on_task_done(self, run_id: str) -> None:
        self._tasks.pop(run_id, None)
        self._claims.pop(run_id, None)
        self._receipts.pop(run_id, None)
        # 槽位在这里归还：确认之后才释放，下一次拉取发生在上一条结束之后。
        self._slots.release()

    async def shutdown(self) -> None:
        self._closed = True
        background = [task for task in (self._receive_task,) if task is not None]
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
        # 关机的行一律不确认：让重投把它们交给下一个进程，从 checkpoint 续跑。
        for _run_id, _task, work in live:
            if work is not None:
                try:
                    await self._store.release(
                        work, status="interrupted", terminal_reason="server_shutdown"
                    )
                except OperationalError:
                    # 关机释放是尽力而为：写锁超时时该行留在 running，
                    # 可见性到期后由下一个进程接管，锁等待超时不中断关机流程。
                    logger.warning(
                        "worker_release_lock_timeout run_id=%s", work.run_id, exc_info=True
                    )
