"""Opt-in PostgreSQL integration tests for distributed claim semantics."""

import asyncio
import os

import pytest
from sqlalchemy import delete

from deepsearch_agent.observability.tracing.context import new_id
from deepsearch_agent.service.db import make_engine, make_session_factory, migrate_database
from deepsearch_agent.service.event_store import RunEventStore
from deepsearch_agent.service.models import Run, User
from deepsearch_agent.service.notifier import EventNotifier
from deepsearch_agent.service.queue import PostgresRunQueue, RunWork
from deepsearch_agent.service.settings import get_service_config

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        os.getenv("RUN_POSTGRES_INTEGRATION") != "1",
        reason="set RUN_POSTGRES_INTEGRATION=1 to use the configured PostgreSQL",
    ),
]


async def test_two_postgres_claimers_cannot_own_the_same_run():
    database_url = get_service_config().database_url
    assert database_url.startswith("postgresql")
    await migrate_database(database_url)
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

        sender = EventNotifier()
        receiver = EventNotifier()
        await sender.start(database_url)
        await receiver.start(database_url)
        notify_key, notification = receiver.subscribe(run_id)
        try:
            batches = await asyncio.gather(
                RunEventStore(factory, notifier=sender).append(
                    run_id, [{"event_type": "pg-event-a", "payload": {}}]
                ),
                RunEventStore(factory, notifier=sender).append(
                    run_id, [{"event_type": "pg-event-b", "payload": {}}]
                ),
            )
            assert sorted(batch[0]["seq"] for batch in batches) == [1, 2]
            await asyncio.wait_for(notification.get(), timeout=2)
        finally:
            receiver.unsubscribe(run_id, notify_key)
            await sender.close()
            await receiver.close()
    finally:
        async with factory() as session:
            await session.execute(delete(User).where(User.email == email))
            await session.commit()
        await engine.dispose()
