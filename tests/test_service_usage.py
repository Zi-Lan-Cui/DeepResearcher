import asyncio
from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

import pytest
import pytest_asyncio
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import select

from deepsearch_agent.config import LLMConfig
from deepsearch_agent.service.persistence.database import init_db, make_engine, make_session_factory
from deepsearch_agent.service.persistence.models import Run, RunUsage, User
from deepsearch_agent.service.usage import (
    CapacityGate,
    ProviderRateLimiter,
    RunUsageCallback,
    UsageBudgetExceeded,
    UsageRuntime,
    UsageStore,
    bind_usage_runtime,
    record_external_request,
    reset_usage_runtime,
)

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def usage_db(tmp_path):
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path / 'usage.db'}")
    await init_db(engine)
    sessions = make_session_factory(engine)
    async with sessions() as session:
        user = User(email="usage@example.com", password_hash="x")
        session.add(user)
        await session.flush()
        session.add(Run(id="run-usage", user_id=user.id, query="q"))
        await session.commit()
    try:
        yield sessions
    finally:
        await engine.dispose()


async def test_usage_store_records_details_and_aggregates(usage_db):
    store = UsageStore(usage_db)
    config = LLMConfig(
        model="test-model",
        input_usd_per_million=1.0,
        output_usd_per_million=2.0,
        cached_input_usd_per_million=0.5,
        price_version="test-v1",
    )
    callback = RunUsageCallback(
        run_id="run-usage",
        store=store,
        gate=CapacityGate(1),
        rate_limiter=ProviderRateLimiter(requests_per_minute=0, tokens_per_minute=0),
        config=config,
    )
    callback_run_id = uuid4()
    await callback.on_chat_model_start(
        {},
        [[HumanMessage(content="prompt")]],
        run_id=callback_run_id,
        metadata={"langgraph_node": "router", "ls_provider": "fake"},
    )
    response = SimpleNamespace(
        generations=[
            [
                SimpleNamespace(
                    text="answer",
                    message=AIMessage(
                        content="answer",
                        usage_metadata={
                            "input_tokens": 100,
                            "output_tokens": 20,
                            "total_tokens": 120,
                            "input_token_details": {"cache_read": 40},
                        },
                    ),
                )
            ]
        ],
        llm_output={},
    )
    await callback.on_llm_end(response, run_id=callback_run_id)

    token = bind_usage_runtime(UsageRuntime("run-usage", store, config))
    try:
        await record_external_request(category="search", status="completed", duration_ms=12)
    finally:
        reset_usage_runtime(token)

    async with usage_db() as session:
        run = await session.get(Run, "run-usage")
        records = (await session.scalars(select(RunUsage).order_by(RunUsage.created_at))).all()
    assert run is not None
    assert (run.llm_call_count, run.input_tokens, run.output_tokens) == (1, 100, 20)
    assert run.cached_input_tokens == 40
    assert run.external_request_count == 1
    assert run.peak_llm_concurrency == 1
    assert Decimal(run.estimated_cost_usd) == Decimal("0.00012000")
    assert [(record.category, record.status) for record in records] == [
        ("llm", "completed"),
        ("search", "completed"),
    ]
    assert records[0].usage_estimated is False
    assert records[0].detail_json["price_version"] == "test-v1"


async def test_usage_budget_blocks_the_next_model_request(usage_db):
    store = UsageStore(usage_db)
    await store.record(
        run_id="run-usage",
        category="llm",
        component="router",
        status="completed",
        input_tokens=8,
        output_tokens=2,
    )
    with pytest.raises(UsageBudgetExceeded, match="run_token_budget_exhausted"):
        await store.enforce_budget("run-usage", LLMConfig(max_tokens_per_run=10))


async def test_daily_user_cost_budget_uses_usage_details(usage_db):
    store = UsageStore(usage_db)
    await store.record(
        run_id="run-usage",
        category="llm",
        component="router",
        status="completed",
        cost_usd=Decimal("0.25"),
    )
    with pytest.raises(UsageBudgetExceeded, match="user_daily_cost_budget_exhausted"):
        await store.enforce_budget("run-usage", LLMConfig(max_cost_usd_per_user_daily=0.2))


async def test_callback_checks_budget_for_every_provider_attempt(usage_db):
    store = UsageStore(usage_db)
    await store.record(
        run_id="run-usage",
        category="llm",
        component="router",
        status="completed",
        input_tokens=10,
    )
    callback = RunUsageCallback(
        run_id="run-usage",
        store=store,
        gate=CapacityGate(1),
        rate_limiter=ProviderRateLimiter(requests_per_minute=0, tokens_per_minute=0),
        config=LLMConfig(max_tokens_per_run=10),
    )
    with pytest.raises(UsageBudgetExceeded, match="run_token_budget_exhausted"):
        await callback.on_chat_model_start({}, [[HumanMessage(content="next")]], run_id=uuid4())


async def test_capacity_and_rate_limiters_bound_requests():
    gate = CapacityGate(1)
    await gate.acquire()
    waiting = asyncio.create_task(gate.acquire())
    await asyncio.sleep(0)
    assert not waiting.done()
    await gate.release()
    await asyncio.wait_for(waiting, timeout=0.1)
    await gate.release()

    limiter = ProviderRateLimiter(
        requests_per_minute=1,
        tokens_per_minute=0,
        window_seconds=0.02,
    )
    await limiter.acquire(1)
    started = asyncio.get_running_loop().time()
    await limiter.acquire(1)
    assert asyncio.get_running_loop().time() - started >= 0.015
