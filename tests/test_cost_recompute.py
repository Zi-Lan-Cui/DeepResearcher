from decimal import Decimal

import pytest
import pytest_asyncio

from deepresearcher import pricing
from deepresearcher.pricing import PRICE_TABLE_VERSION
from deepresearcher.service.cost_recompute import recompute_llm_costs
from deepresearcher.service.persistence.database import init_db, make_engine, make_session_factory
from deepresearcher.service.persistence.models import Run, RunUsage, User

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def recompute_db(tmp_path):
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path / 'recompute.db'}")
    await init_db(engine)
    factory = make_session_factory(engine)
    async with factory() as session:
        user = User(email="recompute@example.com", password_hash="x")
        session.add(user)
        await session.flush()
        session.add(Run(id="run-a", user_id=user.id, query="q"))
        session.add_all(
            [
                RunUsage(
                    id="u-actual",
                    run_id="run-a",
                    category="llm",
                    component="router",
                    status="completed",
                    model="unit-mimo-x",
                    input_tokens=1_000_000,
                    output_tokens=100_000,
                    cached_input_tokens=200_000,
                    detail_json={"price_version": "unpriced"},
                ),
                RunUsage(
                    id="u-estimated",
                    run_id="run-a",
                    category="llm",
                    component="router",
                    status="failed",
                    model="unit-mimo-x",
                    input_tokens=5_000,
                    output_tokens=0,
                    cached_input_tokens=0,
                    usage_estimated=True,
                    detail_json={"price_version": "unpriced"},
                ),
            ]
        )
        await session.commit()
    try:
        yield factory
    finally:
        await engine.dispose()


async def test_recompute_prices_rows_once_and_is_idempotent(recompute_db, monkeypatch):
    monkeypatch.setitem(
        pricing.MODEL_PRICES,
        "unit-mimo-x",
        pricing.ModelPrice(
            input_cny_per_million=1.0, cached_input_cny_per_million=0.5, output_cny_per_million=2.0
        ),
    )
    # 期望：未命中 800_000×1.0 + 命中 200_000×0.5 + 输出 100_000×2.0 = 1_100_000 元/百万 → ¥1.1。
    assert await recompute_llm_costs(recompute_db) == {"rows_updated": 1, "runs_updated": 1}

    async with recompute_db() as session:
        actual = await session.get(RunUsage, "u-actual")
        estimated = await session.get(RunUsage, "u-estimated")
        run = await session.get(Run, "run-a")
    assert Decimal(str(actual.cost_cny)) == Decimal("1.10000000")
    assert actual.detail_json["price_version"] == PRICE_TABLE_VERSION
    # estimated 行与运行时判定一致：豁免不计入成本。
    assert Decimal(str(estimated.cost_cny)) == Decimal("0")
    assert estimated.detail_json["price_version"] == "unpriced"
    assert Decimal(str(run.estimated_cost_cny)) == Decimal("1.10000000")

    # 再跑一遍不动任何行：期望版本与现价一致即幂等界。
    assert await recompute_llm_costs(recompute_db) == {"rows_updated": 0, "runs_updated": 0}


async def test_recompute_keeps_unlisted_model_honestly_unpriced(recompute_db):
    """缺价模型不反复触碰：期望版本就是 unpriced，重跑应零改写。"""
    assert await recompute_llm_costs(recompute_db) == {"rows_updated": 0, "runs_updated": 0}
    async with recompute_db() as session:
        actual = await session.get(RunUsage, "u-actual")
    assert actual.detail_json["price_version"] == "unpriced"
