"""分布式领取语义的 PostgreSQL 集成测试,需显式开启。"""

import asyncio
import os
from dataclasses import replace

import pytest
import pytest_asyncio
from sqlalchemy import delete, update

from deepresearcher.observability.span import new_id
from deepresearcher.service.coordination.signals import PostgresSignalBus
from deepresearcher.service.events.buffer import RunEventBuffer
from deepresearcher.service.events.store import RunEventStore
from deepresearcher.service.runs.admission import QuotaExceededError, RunAdmission
from deepresearcher.service.runs.store import RunStore, RunWork
from deepresearcher.service.settings import get_service_config
from deepresearcher.service.storage.engine import (
    make_engine,
    make_session_factory,
    migrate_database,
)
from deepresearcher.service.storage.tables import Run, User
from deepresearcher.service.storage.tables import utcnow as _utcnow
from messaging_stub import InProcessDispatchBus
from pg_backend import reset_all

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        os.getenv("RUN_POSTGRES_INTEGRATION") != "1",
        reason="set RUN_POSTGRES_INTEGRATION=1 to use the configured PostgreSQL",
    ),
]


@pytest_asyncio.fixture(autouse=True)
async def _clean_schema():
    """这组用例直连服务库、不经过 pg_backend 的按用例清表，留别处的行会撞主键。"""
    reset_all(get_service_config().database_url)


async def test_postgres_service_signals_cross_connections():
    database_url = get_service_config().database_url
    assert database_url.startswith("postgresql")
    sender = PostgresSignalBus()
    receiver = PostgresSignalBus()
    cancel_received = asyncio.Event()
    await sender.start(database_url)
    await receiver.start(database_url)
    cancel_key = receiver.subscribe(
        "run_cancel_requested",
        lambda run_id: cancel_received.set() if run_id == "control-notifier-probe" else None,
    )
    try:
        await sender.notify_cancel("control-notifier-probe")
        await asyncio.wait_for(cancel_received.wait(), timeout=1)
    finally:
        receiver.unsubscribe("run_cancel_requested", cancel_key)
        await receiver.close()
        await sender.close()


async def test_two_api_admission_services_share_postgres_quota_lock():
    database_url = get_service_config().database_url
    assert database_url.startswith("postgresql")
    await migrate_database(database_url)
    engine = make_engine(database_url)
    factory = make_session_factory(engine)
    email = f"{new_id('admission-pg-test')}@test.invalid"
    try:
        async with factory() as session:
            user = User(email=email, password_hash="integration-test")
            session.add(user)
            await session.commit()
            user_id = user.id

        config = replace(
            get_service_config(),
            max_concurrent_runs_per_user=1,
            max_global_queued_runs=100,
        )
        # 这里只比两个准入服务的配额裁决，消息投进内存总线后没有消费者也无所谓。
        publisher = InProcessDispatchBus()
        first, second = await asyncio.gather(
            RunAdmission(session_factory=factory, config=config, publisher=publisher).create(
                user_id, "first"
            ),
            RunAdmission(session_factory=factory, config=config, publisher=publisher).create(
                user_id, "second"
            ),
            return_exceptions=True,
        )
        results = (first, second)
        assert sum(isinstance(item, str) for item in results) == 1
        assert sum(isinstance(item, QuotaExceededError) for item in results) == 1
    finally:
        async with factory() as session:
            await session.execute(delete(User).where(User.email == email))
            await session.commit()
        await engine.dispose()


async def test_acquire_advances_fence_and_retires_the_previous_owner():
    """同一条消息再来一次就是接管：fence 前进，旧凭据只剩 lost。

    单租约下消息到达本身证明上一个持有者已停止续期，正确性由 fence 与写权谓词承担。
    """
    database_url = get_service_config().database_url
    assert database_url.startswith("postgresql")
    # API 与多 Worker 可同时起进程；迁移必须跨进程排队。
    await asyncio.gather(migrate_database(database_url), migrate_database(database_url))
    engine = make_engine(database_url)
    factory = make_session_factory(engine)
    run_id = new_id("run-pg-test")
    email = f"{run_id}@test.invalid"
    try:
        async with factory() as session:
            user = User(email=email, password_hash="integration-test")
            session.add(user)
            await session.flush()
            user_id = user.id
            session.add(Run(id=run_id, user_id=user_id, query="acquire race", status="queued"))
            await session.commit()

        store = RunStore(factory)
        first = await store.acquire(run_id, worker_id="pg-worker-a")
        assert first is not None
        assert first.fence_token == 1
        assert first.resume is False  # queued 首跑
        assert await store.verify_lease(first) == "held"

        takeover = await store.acquire(run_id, worker_id="pg-worker-b")
        assert takeover is not None
        assert takeover.fence_token == 2
        assert takeover.resume is True  # 来源是 running，说明这是接管
        assert await store.verify_lease(takeover) == "held"
        assert await store.verify_lease(first) == "lost"

        # seq 并发安全直测 event_store.append 只负责事务内编号落库，不承担通知。
        event_store = RunEventStore(factory)
        batches = await asyncio.gather(
            event_store.append(run_id, [{"event_type": "pg-event-a", "payload": {}}]),
            event_store.append(run_id, [{"event_type": "pg-event-b", "payload": {}}]),
        )
        assert sorted(batch[0]["seq"] for batch in batches) == [1, 2]

        # committed 唤醒经 buffer.flush 投递，按该路径断言。
        sender = PostgresSignalBus()
        receiver = PostgresSignalBus()
        await sender.start(database_url)
        await receiver.start(database_url)
        notify_key, notification = receiver.subscribe_event(run_id)
        try:
            buffer = RunEventBuffer(store=event_store, signal_bus=sender)
            buffer.open(run_id)
            buffer.write({"run_id": run_id, "event_type": "pg-event-c", "payload": {}})
            await buffer.flush(run_id)
            await asyncio.wait_for(notification.get(), timeout=2)
        finally:
            receiver.unsubscribe("event_committed", notify_key)
            await sender.close()
            await receiver.close()

        # 优雅关停让位后仍是 interrupted，下一个持有者接着续跑。
        assert await store.release(
            takeover, status="interrupted", terminal_reason="integration_test"
        )
        resumed = await store.acquire(run_id, worker_id="pg-worker-c")
        assert resumed is not None
        assert resumed.fence_token == 3
        assert resumed.resume is True
        assert await store.release(
            resumed, status="interrupted", terminal_reason="integration_test"
        )

    finally:
        async with factory() as session:
            await session.execute(delete(User).where(User.email == email))
            await session.commit()
        await engine.dispose()


async def test_acquire_refuses_rows_that_must_not_run():
    """等待澄清、终态、带取消标记的行都不能被领取。"""
    database_url = get_service_config().database_url
    assert database_url.startswith("postgresql")
    await migrate_database(database_url)
    engine = make_engine(database_url)
    factory = make_session_factory(engine)
    try:
        async with factory() as session:
            session.add(User(id=1, email="acquire-refuse@test.invalid", password_hash="h"))
            await session.flush()
            session.add_all(
                [
                    Run(id="run-awaiting", user_id=1, query="q", status="awaiting_input"),
                    Run(id="run-done", user_id=1, query="q", status="completed"),
                    Run(
                        id="run-cancelled-flag",
                        user_id=1,
                        query="q",
                        status="queued",
                        cancellation_requested_at=_utcnow(),
                    ),
                ]
            )
            await session.commit()

        store = RunStore(factory)
        assert await store.acquire("run-awaiting", worker_id="w") is None
        assert await store.acquire("run-done", worker_id="w") is None
        assert await store.acquire("run-cancelled-flag", worker_id="w") is None
        assert await store.acquire("run-missing", worker_id="w") is None

        # 取消标记撤掉后才能领：谓词是「带标记就一律不领」。
        async with factory() as session:
            await session.execute(
                update(Run)
                .where(Run.id == "run-cancelled-flag")
                .values(cancellation_requested_at=None)
            )
            await session.commit()
        work = await store.acquire("run-cancelled-flag", worker_id="w")
        assert work is not None
        assert isinstance(work, RunWork)
    finally:
        await engine.dispose()
