"""跨 worker 共享冷却位；及 SearchService 命中账户级不可用即写入。"""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from deepresearcher.config import SearchConfig
from deepresearcher.health import Cooldown, HealthLedger, MemoryHealthLedger
from deepresearcher.service.execution.provider_health import ProviderHealthStore
from deepresearcher.service.storage.engine import make_engine, make_session_factory
from deepresearcher.service.storage.tables import ProviderHealthRecord
from deepresearcher.tools.errors import ProviderExhaustedError
from deepresearcher.tools.transport import HttpClient
from deepresearcher.tools.web.search.service import SearchService
from pg_backend import database_url_for

pytestmark = pytest.mark.asyncio

KEY = "search:tavily"


def _factory(tmp_path):
    engine = make_engine(database_url_for(tmp_path))
    return engine, make_session_factory(engine)


async def test_trip_then_unavailable_and_expiry(tmp_path):
    engine, factory = _factory(tmp_path)
    health = ProviderHealthStore(factory)
    assert await health.unavailable(KEY) is None
    await health.trip(KEY, "quota_exhausted", 3600)
    blocked = await health.unavailable(KEY)
    assert blocked is not None
    assert blocked.reason == "quota_exhausted"
    assert 3000 < blocked.retry_after <= 3600
    # 过期后自动视为可用：直接把行内恢复时刻改成过去来触发判定。
    async with factory() as session:
        row = await session.scalar(
            select(ProviderHealthRecord).where(ProviderHealthRecord.provider == KEY)
        )
        row.open_until = datetime.now(timezone.utc) - timedelta(seconds=1)
        await session.commit()
    assert await health.unavailable(KEY) is None
    await engine.dispose()


async def test_trip_only_extends_window(tmp_path):
    engine, factory = _factory(tmp_path)
    health = ProviderHealthStore(factory)
    await health.trip(KEY, "quota_exhausted", 3600)
    await health.trip(KEY, "invalid_key", 1)
    blocked = await health.unavailable(KEY)
    assert blocked is not None and blocked.reason == "quota_exhausted"
    await engine.dispose()


async def test_concurrent_first_trips_survive_pk_race(tmp_path):
    """两 worker 同时首撞同一 key：后提交方重试走 update 分支，冷却写入不得失败。"""
    engine, factory = _factory(tmp_path)
    health = ProviderHealthStore(factory)

    await asyncio.gather(
        health.trip(KEY, "quota_exhausted", 3600),
        health.trip(KEY, "invalid_key", 10),
    )

    blocked = await health.unavailable(KEY)
    assert blocked is not None and blocked.reason == "quota_exhausted"
    await engine.dispose()


async def test_record_failure_cools_down_at_threshold(tmp_path):
    """连续失败到阈值才进冷却；中间成功一次就清零。"""
    engine, factory = _factory(tmp_path)
    health = ProviderHealthStore(factory)
    policy = Cooldown(threshold=3, base_seconds=60, max_seconds=300)

    await health.record_failure(KEY, "tool_request", policy)
    await health.record_failure(KEY, "tool_request", policy)
    assert await health.unavailable(KEY) is None

    await health.record_success(KEY)
    await health.record_failure(KEY, "tool_request", policy)
    await health.record_failure(KEY, "tool_request", policy)
    assert await health.unavailable(KEY) is None  # 已清零，重新计数

    await health.record_failure(KEY, "tool_request", policy)
    blocked = await health.unavailable(KEY)
    assert blocked is not None and blocked.reason == "tool_request"
    assert 55 < blocked.retry_after <= 60
    await engine.dispose()


async def test_strikes_increase_cooldown_and_clear_on_success(tmp_path):
    """一直被踢时冷却按次数递增，恢复一次即归零。"""
    engine, factory = _factory(tmp_path)
    health = ProviderHealthStore(factory)
    policy = Cooldown(threshold=1, base_seconds=60, max_seconds=300)

    timestamps = []
    for _ in range(2):
        await health.record_failure(KEY, "tool_request", policy)
        blocked = await health.unavailable(KEY)
        assert blocked is not None
        timestamps.append(blocked.retry_after)
        await health.record_success(KEY)
    # record_success 清零 strikes，所以两次都回到基准时长而不是递增。
    assert timestamps[0] <= 60 and timestamps[1] <= 60

    for expected in (60, 120, 180):
        await health.record_failure(KEY, "tool_request", policy)
        blocked = await health.unavailable(KEY)
        assert blocked is not None
        assert expected - 5 < blocked.retry_after <= expected
    await engine.dispose()


async def test_store_satisfies_ledger_protocol(tmp_path):
    engine, factory = _factory(tmp_path)
    assert isinstance(ProviderHealthStore(factory), HealthLedger)
    assert isinstance(MemoryHealthLedger(), HealthLedger)
    await engine.dispose()


async def test_searchservice_trips_shared_health_on_provider_exhausted(tmp_path):
    """撞一次账户级不可用即写共享冷却位，下一次搜索在出网前被短路。"""
    engine, factory = _factory(tmp_path)
    health = ProviderHealthStore(factory)
    config = SearchConfig(tavily_api_key="k", provider="tavily")
    http = HttpClient(config)
    client = SearchService(config, http, provider_health=health)

    calls = 0

    class FakeProvider:
        async def asearch(self, query, limit):
            nonlocal calls
            calls += 1
            raise ProviderExhaustedError("quota_exhausted", "HTTP 432")

    client._providers = [("tavily", FakeProvider())]
    with pytest.raises(ProviderExhaustedError):
        await client.asearch("q")
    blocked = await health.unavailable(KEY)
    assert blocked is not None and blocked.reason == "quota_exhausted"

    # 第二次调用的异常来自冷却短路，provider 不会被调到。
    with pytest.raises(ProviderExhaustedError):
        await client.asearch("q2")
    assert calls == 1
    await http.aclose()
    await engine.dispose()


async def test_memory_ledger_trips():
    health = MemoryHealthLedger()
    assert await health.unavailable(KEY) is None
    await health.trip(KEY, "invalid_key", 60)
    blocked = await health.unavailable(KEY)
    assert blocked is not None and blocked.reason == "invalid_key"
