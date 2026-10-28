"""入站死信与取消结算：消息该不该确认、行该不该结案。

分两段看：前半段是不依赖数据库的消费者判定（该不该 ACK、该不该抢执行权），
后半段必须在真实 PostgreSQL 上验证——取消优先与 fence 自增都是条件 UPDATE 的
语义，用假对象验证等于什么都没验证。
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from sqlalchemy import select, update

from deepresearcher.service.execution.consumer import RunConsumer
from deepresearcher.service.messaging.dispatch import new_dispatch
from deepresearcher.service.messaging.receiver import DispatchDelivery
from deepresearcher.service.runs.store import RunStateView, RunStore, RunWork, TerminalState
from deepresearcher.service.storage.engine import make_engine, make_session_factory
from deepresearcher.service.storage.tables import Run, User
from pg_backend import database_url_for

pytestmark = pytest.mark.asyncio

NOW = datetime(2026, 10, 24, 12, 0, 0, tzinfo=timezone.utc)


def _inbound(run_id: str, attempts: int) -> DispatchDelivery:
    return DispatchDelivery(
        message=new_dispatch(run_id), receipt=object(), delivery_attempt=attempts
    )


class _RecordingReceiver:
    def __init__(self) -> None:
        self.acked: list[str] = []
        self.extended: list[tuple[str, int]] = []

    async def receive(self) -> None:
        return None

    async def ack(self, delivery: DispatchDelivery) -> None:
        self.acked.append(delivery.message.run_id)

    async def extend_visibility(self, delivery: DispatchDelivery, seconds: int) -> None:
        self.extended.append((delivery.message.run_id, seconds))

    async def close(self) -> None:
        return None


class _RecordingBuffer:
    """记录 done 帧的四段式调用：open → write_done → flush → close。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.done_payloads: list[dict] = []

    def open(self, run_id: str) -> None:
        self.calls.append(("open", run_id))

    def write_done(self, run_id: str, **payload) -> None:
        self.calls.append(("done", run_id))
        self.done_payloads.append(payload)

    async def flush(self, run_id: str) -> None:
        self.calls.append(("flush", run_id))

    def close(self, run_id: str) -> None:
        self.calls.append(("close", run_id))


class _DispatchStore:
    """只回答「这行现在什么状态」，并把结算与领取调用记下来。"""

    def __init__(
        self,
        state: RunStateView | None,
        *,
        settled: str | None = "failed",
        settle_error: Exception | None = None,
        cancel_settled: bool = True,
        cancel_error: Exception | None = None,
    ) -> None:
        self._state = state
        self._settled = settled
        self._settle_error = settle_error
        self._cancel_settled = cancel_settled
        self._cancel_error = cancel_error
        self.settle_calls: list[str] = []
        self.cancel_calls: list[str] = []
        self.acquire_calls: list[str] = []

    async def inspect(self, run_id: str) -> RunStateView | None:
        return self._state

    async def settle_exhausted(self, run_id: str) -> str | None:
        if self._settle_error is not None:
            raise self._settle_error
        self.settle_calls.append(run_id)
        return self._settled

    async def settle_cancelled(self, run_id: str) -> TerminalState | None:
        if self._cancel_error is not None:
            raise self._cancel_error
        self.cancel_calls.append(run_id)
        if not self._cancel_settled:
            return None
        return TerminalState(status="cancelled", answer_mode="", report_available=False)

    async def acquire(self, run_id: str, *, worker_id: str) -> RunWork | None:
        self.acquire_calls.append(run_id)
        return None


def _consumer(
    store: object,
    receiver: _RecordingReceiver,
    *,
    max_attempts: int = 20,
    buffer: object | None = None,
) -> RunConsumer:
    return RunConsumer(
        store=store,  # type: ignore[arg-type] - 鸭子类型的判定替身
        executor=object(),  # type: ignore[arg-type] - 死信路径不碰执行器
        buffer=buffer or object(),  # type: ignore[arg-type] - 死信路径不写事件
        receiver=receiver,
        max_running=1,
        lease_check_seconds=15,
        invisible_seconds=90,
        visibility_renew_seconds=30,
        max_attempts=max_attempts,
    )


async def test_awaiting_input_dispatch_is_acked_without_settling():
    """等待澄清的行：这条消息已经没用了，确认掉即可。

    回答会走 answer_resume 另发一条消息；不确认的话，这条旧消息会一直重投，
    最后被自己的死信判据结案——把一个正常的暂停状态误判成投递故障。
    """
    store = _DispatchStore(RunStateView("awaiting_input", None))
    receiver = _RecordingReceiver()
    consumer = _consumer(store, receiver, max_attempts=20)

    await consumer._accept(_inbound("run-paused", attempts=1))  # noqa: SLF001

    assert receiver.acked == ["run-paused"]
    assert store.acquire_calls == []
    assert store.settle_calls == []


async def test_cancelled_running_row_is_settled_and_acked():
    """取消标记 + running：消息能到达就证明原持有者已停止续期，就地结算并发 done 帧。

    这条路径不能等执行器自己收尾——它可能已经死了；也不能不确认，
    否则消息会一直重投到投递上限。
    """
    store = _DispatchStore(RunStateView("running", NOW))
    receiver = _RecordingReceiver()
    buffer = _RecordingBuffer()
    consumer = _consumer(store, receiver, buffer=buffer)

    await consumer._accept(_inbound("run-cancel", attempts=1))  # noqa: SLF001

    assert store.cancel_calls == ["run-cancel"]
    assert store.acquire_calls == []
    assert receiver.acked == ["run-cancel"]
    assert buffer.calls == [
        ("open", "run-cancel"),
        ("done", "run-cancel"),
        ("flush", "run-cancel"),
        ("close", "run-cancel"),
    ]


async def test_cancel_settle_failure_keeps_message():
    """结算写不进库就不确认：取消意图还在，下一次投递重判即可。

    早一步确认会让这一行既不是终态、也没有任何消息指向它——单租约下没有扫描器兜底。
    """
    store = _DispatchStore(RunStateView("running", NOW), cancel_error=RuntimeError("db down"))
    receiver = _RecordingReceiver()
    consumer = _consumer(store, receiver, buffer=_RecordingBuffer())

    await consumer._accept(_inbound("run-cancel-retry", attempts=1))  # noqa: SLF001

    assert receiver.acked == []


async def test_exhausted_dispatch_settles_and_acks_without_acquiring():
    """投递次数用尽：结案 + 确认，且绝不抢执行权。"""
    store = _DispatchStore(RunStateView("queued", None))
    receiver = _RecordingReceiver()
    consumer = _consumer(store, receiver, max_attempts=3)

    await consumer._accept(_inbound("run-exhausted", attempts=3))  # noqa: SLF001

    assert store.settle_calls == ["run-exhausted"]
    assert store.acquire_calls == []
    assert receiver.acked == ["run-exhausted"]


async def test_exhausted_dispatch_keeps_message_when_settle_misses():
    """结案没命中就不确认：消息是这一行唯一的路标。

    行可能已被别人结算（那一刻它已是终态，下次投递会走终态分支确认掉），
    也可能写库失败——两种情况都不能靠「猜」丢掉这条消息。
    """
    store = _DispatchStore(RunStateView("running", None), settled=None)
    receiver = _RecordingReceiver()
    consumer = _consumer(store, receiver, max_attempts=3)

    await consumer._accept(_inbound("run-busy", attempts=3))  # noqa: SLF001

    assert store.settle_calls == ["run-busy"]
    assert receiver.acked == []


async def test_exhausted_dispatch_keeps_message_when_settle_fails():
    """写不进数据库就不确认：这条消息还得靠重投再来一次。"""
    store = _DispatchStore(RunStateView("queued", None), settle_error=RuntimeError("db down"))
    receiver = _RecordingReceiver()
    consumer = _consumer(store, receiver, max_attempts=3)

    await consumer._accept(_inbound("run-retry", attempts=3))  # noqa: SLF001

    assert receiver.acked == []


async def test_healthy_dispatch_below_ceiling_tries_to_acquire():
    """未到上限时直接抢执行权：不确认，也不推可见性（让位分支已随双租约删除）。"""
    store = _DispatchStore(RunStateView("running", None))
    receiver = _RecordingReceiver()
    consumer = _consumer(store, receiver, max_attempts=20)

    await consumer._accept(_inbound("run-live", attempts=2))  # noqa: SLF001

    assert store.settle_calls == []
    assert store.cancel_calls == []
    assert store.acquire_calls == ["run-live"]
    assert receiver.acked == []
    assert receiver.extended == []


# ---------------------------------------------------------------------------
# 以下用例需要真实 PostgreSQL：条件 UPDATE 的语义只能在那里验证。
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def store_env(tmp_path):
    engine = make_engine(database_url_for(tmp_path))
    factory = make_session_factory(engine)
    async with factory() as session:
        session.add(User(id=1, email="dlq@test", password_hash="h"))
        await session.commit()
    try:
        yield RunStore(factory), factory
    finally:
        await engine.dispose()


async def _seed(factory, run_id: str, **values) -> None:
    async with factory() as session:
        session.add(Run(id=run_id, user_id=1, query="q", **values))
        await session.commit()


async def _row(factory, run_id: str) -> Run:
    async with factory() as session:
        row = await session.scalar(select(Run).where(Run.id == run_id))
    assert row is not None
    return row


async def test_settle_exhausted_fails_queued_run(store_env):
    store, factory = store_env
    await _seed(factory, "run-1", status="queued")

    assert await store.settle_exhausted("run-1") == "failed"

    row = await _row(factory, "run-1")
    assert row.status == "failed"
    assert row.terminal_reason == "dispatch_attempts_exhausted"
    assert row.finished_at is not None
    assert row.lease_owner is None


async def test_settle_exhausted_fails_running_row_without_owner(store_env):
    """投递耗尽也能结掉 running：二十次可见性到期都没续上，持有者已经不在了。"""
    store, factory = store_env
    await _seed(factory, "run-3", status="running", lease_owner="worker-a", fence_token=1)

    assert await store.settle_exhausted("run-3") == "failed"

    row = await _row(factory, "run-3")
    assert row.status == "failed"
    assert row.terminal_reason == "dispatch_attempts_exhausted"
    assert row.lease_owner is None


async def test_settle_exhausted_prefers_cancellation(store_env):
    """取消优先：带取消标记的行落 cancelled，不落 failed。"""
    store, factory = store_env
    await _seed(factory, "run-2", status="queued", cancellation_requested_at=NOW)

    assert await store.settle_exhausted("run-2") == "cancelled"

    row = await _row(factory, "run-2")
    assert row.status == "cancelled"
    assert row.terminal_reason == "user_cancelled"


async def test_settle_exhausted_prefers_cancellation_on_running(store_env):
    """带取消标记的 running 行也必须落 cancelled：它两个分支都要覆盖到。"""
    store, factory = store_env
    await _seed(
        factory,
        "run-3b",
        status="running",
        lease_owner="worker-a",
        fence_token=1,
        cancellation_requested_at=NOW,
    )

    assert await store.settle_exhausted("run-3b") == "cancelled"

    row = await _row(factory, "run-3b")
    assert row.status == "cancelled"
    assert row.terminal_reason == "user_cancelled"


async def test_settle_exhausted_refuses_to_fail_a_cancelling_row(store_env):
    """取消插队到两次写之间时，失败分支必须打空，宁可留给重投。

    取消分支「没命中」这件事只能靠替身制造；被验证的是失败分支那条
    cancellation_requested_at IS NULL 谓词，它跑在真库上。
    """
    store, factory = store_env
    await _seed(
        factory,
        "run-3c",
        status="running",
        lease_owner="worker-a",
        fence_token=1,
        cancellation_requested_at=NOW,
    )

    with patch.object(RunStore, "settle_cancelled", new=AsyncMock(return_value=None)):
        assert await store.settle_exhausted("run-3c") is None

    assert (await _row(factory, "run-3c")).status == "running"


async def test_settle_cancelled_covers_running_row(store_env):
    """单行取消结算同样覆盖 running：消息到达即证明原持有者已停止续期。"""
    store, factory = store_env
    await _seed(
        factory,
        "run-6",
        status="running",
        lease_owner="worker-a",
        fence_token=2,
        cancellation_requested_at=NOW,
    )

    assert await store.settle_cancelled("run-6") is not None

    row = await _row(factory, "run-6")
    assert row.status == "cancelled"
    assert row.terminal_reason == "user_cancelled"
    assert row.lease_owner is None
    # 已终态的行再结算一次不重复命中。
    assert await store.settle_cancelled("run-6") is None


async def test_settle_cancelled_ignores_row_without_flag(store_env):
    store, factory = store_env
    await _seed(factory, "run-7", status="running", lease_owner="worker-a", fence_token=1)

    assert await store.settle_cancelled("run-7") is None

    assert (await _row(factory, "run-7")).status == "running"


async def test_verify_lease_reports_three_verdicts(store_env):
    """租约探针的三态：还持有 / 用户要取消 / 已经不是持有者。"""
    store, factory = store_env
    await _seed(factory, "run-8", status="running", lease_owner="worker-a", fence_token=3)
    work = RunWork(run_id="run-8", user_id=1, query="q", lease_owner="worker-a", fence_token=3)

    assert await store.verify_lease(work) == "held"

    async with factory() as session:
        await session.execute(
            update(Run).where(Run.id == "run-8").values(cancellation_requested_at=NOW)
        )
        await session.commit()
    assert await store.verify_lease(work) == "cancelled"

    # 被接管：fence 前进，旧凭据从此只能看到 lost。
    async with factory() as session:
        await session.execute(update(Run).where(Run.id == "run-8").values(fence_token=4))
        await session.commit()
    assert await store.verify_lease(work) == "lost"


async def test_verify_lease_treats_unclaimed_work_as_lost(store_env):
    store, _factory = store_env
    assert await store.verify_lease(RunWork(run_id="run-none", user_id=1, query="q")) == "lost"


async def test_reopen_exhausted_returns_row_for_redispatch(store_env):
    """显式恢复：失败行退回 interrupted，fence 自增，随后可被正常领取。"""
    store, factory = store_env
    await _seed(factory, "run-4", status="queued")
    assert await store.settle_exhausted("run-4") == "failed"

    assert await store.reopen_exhausted("run-4") is True

    row = await _row(factory, "run-4")
    assert row.status == "interrupted"
    assert row.terminal_reason is None
    assert row.finished_at is None
    assert row.fence_token == 1  # 恢复自己就加一次，防止旧凭据复活

    work = await store.acquire("run-4", worker_id="worker-b")

    assert work is not None
    assert work.fence_token == 2
    assert work.resume is True  # interrupted 意味着跑过一半


async def test_reopen_exhausted_refuses_other_terminal_reasons(store_env):
    """只认投递耗尽这一个原因：真正跑过又失败的行不许被这条路径复活。"""
    store, factory = store_env
    await _seed(factory, "run-5", status="failed", terminal_reason="server_restart", fence_token=1)

    assert await store.reopen_exhausted("run-5") is False

    row = await _row(factory, "run-5")
    assert row.status == "failed"
    assert row.terminal_reason == "server_restart"
