"""跨 worker 共享熔断位（Postgres/SQLite 后端）+ SearchClient 命中即触发的行为。"""

import pytest
from sqlalchemy import select

from deepresearcher.config import SearchConfig
from deepresearcher.service.persistence.database import init_db, make_engine, make_session_factory
from deepresearcher.service.persistence.provider_health import PostgresProviderHealth
from deepresearcher.tools.errors import ProviderExhaustedError
from deepresearcher.tools.transport import HttpClient
from deepresearcher.tools.web.search.client import SearchClient
from deepresearcher.tools.web.search.health import MemoryProviderHealth

pytestmark = pytest.mark.asyncio


def _factory(tmp_path):
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path / 'ph.db'}")
    return engine, make_session_factory(engine)


async def test_trip_then_is_open_and_expiry(tmp_path):
    engine, factory = _factory(tmp_path)
    await init_db(engine)
    health = PostgresProviderHealth(factory)
    assert await health.is_open("tavily") is None
    await health.trip("tavily", "quota_exhausted", 3600)
    assert await health.is_open("tavily") == "quota_exhausted"
    # 到期：直接把恢复时刻改成过去，验证过期后自动视为可用（trip 只会延长，不会缩短）。
    from datetime import datetime, timedelta, timezone

    from deepresearcher.service.persistence.models import ProviderHealthRecord

    async with factory() as session:
        row = await session.scalar(
            select(ProviderHealthRecord).where(ProviderHealthRecord.provider == "tavily")
        )
        row.open_until = datetime.now(timezone.utc) - timedelta(seconds=1)
        await session.commit()
    assert await health.is_open("tavily") is None
    await engine.dispose()


async def test_trip_only_extends_window(tmp_path):
    engine, factory = _factory(tmp_path)
    await init_db(engine)
    health = PostgresProviderHealth(factory)
    await health.trip("tavily", "quota_exhausted", 3600)
    await health.trip("tavily", "invalid_key", 1)  # 更短窗口不得缩短
    assert await health.is_open("tavily") == "quota_exhausted"
    await engine.dispose()


async def test_searchclient_trips_shared_health_on_provider_exhausted(tmp_path):
    """撞一次账户级不可用 → 写共享健康位 → 下一次搜索在出网前就被短路（跨 worker 生效的机制）。"""
    engine, factory = _factory(tmp_path)
    await init_db(engine)
    health = PostgresProviderHealth(factory)
    config = SearchConfig(tavily_api_key="k", provider="tavily")
    http = HttpClient(config)
    client = SearchClient(config, http, provider_health=health)

    async def _boom(query, limit):  # 假冒 provider：抛账户级错误
        raise ProviderExhaustedError("quota_exhausted", "HTTP 432")

    client._provider = lambda: type("P", (), {"asearch": staticmethod(_boom)})()  # type: ignore[method-assign]
    with pytest.raises(ProviderExhaustedError):
        await client.asearch("q")
    assert await health.is_open("tavily") == "quota_exhausted"
    # 二次调用应被熔断短路（provider 甚至不会被调到）
    with pytest.raises(ProviderExhaustedError):
        await client.asearch("q2")
    await http.aclose()
    await engine.dispose()


async def test_memory_health_trips():
    health = MemoryProviderHealth()
    assert await health.is_open("tavily") is None
    await health.trip("tavily", "invalid_key", 60)
    assert await health.is_open("tavily") == "invalid_key"
