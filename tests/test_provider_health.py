"""跨 worker 共享熔断位,后端为 Postgres/SQLite;及 SearchService 命中账户级不可用即熔断。"""

import pytest
from sqlalchemy import select

from deepresearcher.config import SearchConfig
from deepresearcher.service.persistence.database import init_db, make_engine, make_session_factory
from deepresearcher.service.persistence.provider_health import PostgresProviderHealth
from deepresearcher.tools.errors import ProviderExhaustedError
from deepresearcher.tools.transport import HttpClient
from deepresearcher.tools.web.search.health import MemoryProviderHealth
from deepresearcher.tools.web.search.service import SearchService

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
    # 过期后自动视为可用:直接把行内恢复时刻改成过去来触发判定。
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
    await health.trip("tavily", "invalid_key", 1)
    assert await health.is_open("tavily") == "quota_exhausted"
    await engine.dispose()


async def test_concurrent_first_trips_survive_pk_race(tmp_path):
    """两 worker 同时首撞同一 provider:后提交方重试走 update 分支,不得让熔断写入失败转成主业务异常。"""
    import asyncio

    engine, factory = _factory(tmp_path)
    await init_db(engine)
    health = PostgresProviderHealth(factory)

    await asyncio.gather(
        health.trip("tavily", "quota_exhausted", 3600),
        health.trip("tavily", "invalid_key", 10),
    )

    # 赢家写谁都可能，但只有一条记录、窗口取更长者、不抛异常。
    assert await health.is_open("tavily") == "quota_exhausted"
    await engine.dispose()


async def test_searchservice_trips_shared_health_on_provider_exhausted(tmp_path):
    """撞一次账户级不可用即写共享熔断位,下一次搜索在出网前被短路,机制跨 worker 生效。"""
    engine, factory = _factory(tmp_path)
    await init_db(engine)
    health = PostgresProviderHealth(factory)
    config = SearchConfig(tavily_api_key="k", provider="tavily")
    http = HttpClient(config)
    client = SearchService(config, http, provider_health=health)

    async def _boom(query, limit):
        raise ProviderExhaustedError("quota_exhausted", "HTTP 432")

    client._provider = lambda: type("P", (), {"asearch": staticmethod(_boom)})()  # type: ignore[method-assign]
    with pytest.raises(ProviderExhaustedError):
        await client.asearch("q")
    assert await health.is_open("tavily") == "quota_exhausted"
    # 第二次调用的异常来自熔断短路,provider 不会被调到。
    with pytest.raises(ProviderExhaustedError):
        await client.asearch("q2")
    await http.aclose()
    await engine.dispose()


async def test_memory_health_trips():
    health = MemoryProviderHealth()
    assert await health.is_open("tavily") is None
    await health.trip("tavily", "invalid_key", 60)
    assert await health.is_open("tavily") == "invalid_key"
