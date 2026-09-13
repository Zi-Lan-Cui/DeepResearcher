"""Opt-in PostgreSQL integration tests for distributed claim semantics."""

import asyncio
import os
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import delete, update

from deepresearcher.observability.tracing.context import new_id
from deepresearcher.service.events.store import RunEventStore
from deepresearcher.service.persistence.database import (
    make_engine,
    make_session_factory,
    migrate_database,
)
from deepresearcher.service.persistence.models import Run, ToolCacheEntry, User
from deepresearcher.service.persistence.tool_cache import PostgresToolCache
from deepresearcher.service.runs.queue import PostgresRunQueue, RunWork
from deepresearcher.service.runs.service import QuotaExceededError, RunService
from deepresearcher.service.settings import get_service_config
from deepresearcher.service.signals import PostgresSignalBus
from deepresearcher.tools.cache import CacheValue

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        os.getenv("RUN_POSTGRES_INTEGRATION") != "1",
        reason="set RUN_POSTGRES_INTEGRATION=1 to use the configured PostgreSQL",
    ),
]


async def test_postgres_service_signals_cross_connections():
    database_url = get_service_config().database_url
    assert database_url.startswith("postgresql")
    sender = PostgresSignalBus()
    receiver = PostgresSignalBus()
    cancel_received = asyncio.Event()
    work_received = asyncio.Event()
    await sender.start(database_url)
    await receiver.start(database_url)
    cancel_key = receiver.subscribe(
        "run_cancel_requested",
        lambda run_id: cancel_received.set() if run_id == "control-notifier-probe" else None,
    )
    work_key = receiver.subscribe("run_available", lambda _payload: work_received.set())
    try:
        await sender.notify_cancel("control-notifier-probe")
        await sender.notify_work_available()
        await asyncio.wait_for(
            asyncio.gather(cancel_received.wait(), work_received.wait()), timeout=1
        )
    finally:
        receiver.unsubscribe("run_cancel_requested", cancel_key)
        receiver.unsubscribe("run_available", work_key)
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
        first, second = await asyncio.gather(
            RunService(session_factory=factory, config=config).create(user_id, "first"),
            RunService(session_factory=factory, config=config).create(user_id, "second"),
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


async def test_two_postgres_claimers_cannot_own_the_same_run():
    database_url = get_service_config().database_url
    assert database_url.startswith("postgresql")
    # API 与多 Worker 可同时起进程；迁移必须跨进程排队。
    await asyncio.gather(migrate_database(database_url), migrate_database(database_url))
    engine = make_engine(database_url)
    factory = make_session_factory(engine)
    run_id = new_id("run-pg-test")
    cache_key = new_id("cache-pg-test")
    email = f"{run_id}@test.invalid"
    try:
        async with factory() as session:
            user = User(email=email, password_hash="integration-test")
            session.add(user)
            await session.flush()
            user_id = user.id
            session.add(Run(id=run_id, user_id=user_id, query="claim race", status="queued"))
            await session.commit()

        preferred = RunWork(run_id=run_id, user_id=user_id, query="claim race")
        first, second = await asyncio.gather(
            PostgresRunQueue(factory).claim(
                worker_id="pg-worker-a", lease_seconds=60, preferred=preferred
            ),
            PostgresRunQueue(factory).claim(
                worker_id="pg-worker-b", lease_seconds=60, preferred=preferred
            ),
        )
        winners = [claim for claim in (first, second) if claim is not None]
        assert len(winners) == 1
        assert winners[0].attempt == 1

        sender = PostgresSignalBus()
        receiver = PostgresSignalBus()
        await sender.start(database_url)
        await receiver.start(database_url)
        notify_key, notification = receiver.subscribe_event(run_id)
        try:
            batches = await asyncio.gather(
                RunEventStore(factory, signal_bus=sender).append(
                    run_id, [{"event_type": "pg-event-a", "payload": {}}]
                ),
                RunEventStore(factory, signal_bus=sender).append(
                    run_id, [{"event_type": "pg-event-b", "payload": {}}]
                ),
            )
            assert sorted(batch[0]["seq"] for batch in batches) == [1, 2]
            await asyncio.wait_for(notification.get(), timeout=2)
        finally:
            receiver.unsubscribe("event_committed", notify_key)
            await sender.close()
            await receiver.close()

        assert await PostgresRunQueue(factory).release(
            winners[0], status="interrupted", terminal_reason="integration_test"
        )
        takeover_run_id = new_id("run-pg-takeover")
        async with factory() as session:
            session.add(
                Run(
                    id=takeover_run_id,
                    user_id=user_id,
                    query="worker crash takeover",
                    status="queued",
                )
            )
            await session.commit()
        takeover_queue = PostgresRunQueue(factory)
        abandoned = await takeover_queue.claim(worker_id="dead-worker", lease_seconds=60)
        assert abandoned is not None
        async with factory() as session:
            await session.execute(
                update(Run)
                .where(Run.id == takeover_run_id)
                .values(lease_expires_at=datetime.now(timezone.utc) - timedelta(seconds=1))
            )
            await session.commit()
        expired = await takeover_queue.reap_expired()
        assert [work.run_id for work in expired] == [takeover_run_id]
        takeover = await PostgresRunQueue(factory).claim(
            worker_id="replacement-worker",
            lease_seconds=60,
            preferred=RunWork(
                run_id=takeover_run_id,
                user_id=user_id,
                query="worker crash takeover",
                resume=True,
            ),
        )
        assert takeover is not None
        assert takeover.lease_owner == "replacement-worker"
        assert takeover.attempt == 2
        assert await takeover_queue.release(
            takeover, status="interrupted", terminal_reason="integration_test"
        )

        second_run_id = new_id("run-pg-slot-a")
        third_run_id = new_id("run-pg-slot-b")
        async with factory() as session:
            session.add_all(
                [
                    Run(id=second_run_id, user_id=user_id, query="slot a", status="queued"),
                    Run(id=third_run_id, user_id=user_id, query="slot b", status="queued"),
                ]
            )
            await session.commit()
        limited_a = PostgresRunQueue(factory, max_global_running=1)
        limited_b = PostgresRunQueue(factory, max_global_running=1)
        slot_claims = await asyncio.gather(
            limited_a.claim(
                worker_id="pg-worker-a",
                lease_seconds=60,
                preferred=RunWork(second_run_id, user_id, "slot a"),
            ),
            limited_b.claim(
                worker_id="pg-worker-b",
                lease_seconds=60,
                preferred=RunWork(third_run_id, user_id, "slot b"),
            ),
        )
        assert sum(claim is not None for claim in slot_claims) == 1

        cache_calls = 0

        async def compute_cache_value():
            nonlocal cache_calls
            cache_calls += 1
            return CacheValue(value={"source": "postgres"})

        first_cache = await PostgresToolCache(factory).get_or_compute(
            "test",
            cache_key,
            ttl_seconds=60,
            schema_version="v1",
            compute=compute_cache_value,
        )
        second_cache = await PostgresToolCache(factory).get_or_compute(
            "test",
            cache_key,
            ttl_seconds=60,
            schema_version="v1",
            compute=compute_cache_value,
        )
        assert first_cache.hit is False and second_cache.hit is True
        assert second_cache.value == {"source": "postgres"}
        assert cache_calls == 1
    finally:
        async with factory() as session:
            await session.execute(
                delete(ToolCacheEntry).where(
                    ToolCacheEntry.namespace == "test",
                    ToolCacheEntry.cache_key == cache_key,
                )
            )
            await session.execute(delete(User).where(User.email == email))
            await session.commit()
        await engine.dispose()
