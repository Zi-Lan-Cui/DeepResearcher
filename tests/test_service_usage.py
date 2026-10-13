import asyncio
from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

import pytest
import pytest_asyncio
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import select

from deepresearcher import pricing
from deepresearcher.config import LLMConfig
from deepresearcher.observability.tracing import spans
from deepresearcher.observability.tracing.context import bind_context, current_span_context
from deepresearcher.service.execution.usage import (
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
from deepresearcher.service.persistence.database import init_db, make_engine, make_session_factory
from deepresearcher.service.persistence.models import Run, RunUsage, User

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


async def test_usage_store_records_details_and_aggregates(usage_db, monkeypatch):
    store = UsageStore(usage_db)
    monkeypatch.setitem(pricing.MODEL_PRICES, "test-model", pricing.ModelPrice(1.0, 0.5, 2.0))
    config = LLMConfig(model="test-model", price_version="test-v1")
    callback = RunUsageCallback(
        run_id="run-usage",
        store=store,
        gate=CapacityGate(1),
        rate_limiter=ProviderRateLimiter(requests_per_minute=0, tokens_per_minute=0),
        config=config,
    )
    callback_run_id = uuid4()
    with spans.trace("research_run", run_id="run-usage") as trace_id:
        with spans.span("router", kind="node"):
            with bind_context(node_id="router"):
                frozen = current_span_context()
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
    assert Decimal(str(run.estimated_cost_cny)) == Decimal("0.00012000")
    assert [(record.category, record.status) for record in records] == [
        ("llm", "completed"),
        ("search", "completed"),
    ]
    assert records[0].usage_estimated is False
    assert records[0].detail_json["price_version"] == "test-v1"
    assert records[0].detail_json["trace_id"] == trace_id == frozen.trace_id
    assert records[0].detail_json["span_id"] == frozen.span_id
    assert records[0].detail_json["node_id"] == "router"


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
        cost_cny=Decimal("0.25"),
    )
    with pytest.raises(UsageBudgetExceeded, match="user_daily_cost_budget_exhausted"):
        await store.enforce_budget("run-usage", LLMConfig(max_cost_cny_per_user_daily=0.2))


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


def _actual_usage_response() -> SimpleNamespace:
    return SimpleNamespace(
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


async def test_unlisted_model_records_zero_cost_and_unpriced_stamp(usage_db):
    """价表未登记的模型：token 照记，成本记 0，版本戳如实落 unpriced。"""
    store = UsageStore(usage_db)
    callback = RunUsageCallback(
        run_id="run-usage",
        store=store,
        gate=CapacityGate(1),
        rate_limiter=ProviderRateLimiter(requests_per_minute=0, tokens_per_minute=0),
        config=LLMConfig(model="unlisted-model-x", price_version="table-v9"),
    )
    run_id = uuid4()
    await callback.on_chat_model_start(
        {}, [[HumanMessage(content="prompt")]], run_id=run_id, metadata={}
    )
    await callback.on_llm_end(_actual_usage_response(), run_id=run_id)

    async with usage_db() as session:
        record = (await session.scalars(select(RunUsage).where(RunUsage.category == "llm"))).one()
        run = await session.get(Run, "run-usage")
    assert record.usage_estimated is False
    assert record.input_tokens == 100
    assert Decimal(str(record.cost_cny)) == Decimal("0")
    assert record.detail_json["price_version"] == "unpriced"
    assert Decimal(str(run.estimated_cost_cny)) == Decimal("0")


async def test_estimated_failure_row_carries_tokens_but_zero_cost(usage_db, monkeypatch):
    """estimated 行 token 是 char/4 估算：即便模型在价表内也不计成本，防伪造数字污染对账。"""
    monkeypatch.setitem(pricing.MODEL_PRICES, "test-model", pricing.ModelPrice(1.0, 0.5, 2.0))
    store = UsageStore(usage_db)
    callback = RunUsageCallback(
        run_id="run-usage",
        store=store,
        gate=CapacityGate(1),
        rate_limiter=ProviderRateLimiter(requests_per_minute=0, tokens_per_minute=0),
        config=LLMConfig(model="test-model", price_version="table-v9"),
    )
    run_id = uuid4()
    await callback.on_chat_model_start(
        {}, [[HumanMessage(content="p" * 400)]], run_id=run_id, metadata={}
    )
    await callback.on_llm_error(RuntimeError("provider down"), run_id=run_id)

    async with usage_db() as session:
        record = (await session.scalars(select(RunUsage).where(RunUsage.category == "llm"))).one()
    assert record.status == "failed"
    assert record.usage_estimated is True
    assert record.input_tokens > 0
    assert Decimal(str(record.cost_cny)) == Decimal("0")
