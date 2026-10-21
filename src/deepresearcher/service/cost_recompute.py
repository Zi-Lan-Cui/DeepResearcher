"""按现行价表重算历史 LLM 用量行的成本，price_version 为幂等边界。

价目更动或补录缺价模型后显式运行一次：
    python -m deepresearcher.service.cost_recompute
estimated 行的 token 是 char/4 估算而非真实回执，与运行时记账按同一判定标准，不计成本，不触碰。
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any

from sqlalchemy import func, select

from deepresearcher.pricing import PRICE_TABLE_VERSION, cost_cny, price_for_model
from deepresearcher.service.persistence.database import make_engine, make_session_factory
from deepresearcher.service.persistence.models import Run, RunUsage
from deepresearcher.service.settings import get_service_config


def _desired_price_version(model: str | None, target_version: str) -> str:
    return target_version if price_for_model(model) else "unpriced"


async def recompute_llm_costs(
    session_factory: Any, *, target_version: str = PRICE_TABLE_VERSION
) -> dict[str, int]:
    """重算 LLM 行的 cost_cny 与 price_version，并按行汇总回写 run 累计成本。

    参数/返回：
        session_factory: 异步会话工厂。
        target_version: 目标价目批次；行的期望版本已是它的跳过不写。
    返回： {"rows_updated": 改写行数， "runs_updated": 回写累计成本的 run 数}。
    """

    rows_updated = 0
    affected_run_ids: set[str] = set()
    async with session_factory() as session:
        usage_rows = (
            await session.scalars(
                select(RunUsage).where(
                    RunUsage.category == "llm", RunUsage.usage_estimated.is_(False)
                )
            )
        ).all()
        for usage_row in usage_rows:
            detail = dict(usage_row.detail_json or {})
            desired_version = _desired_price_version(usage_row.model, target_version)
            new_cost = cost_cny(
                usage_row.model,
                usage_row.input_tokens,
                usage_row.output_tokens,
                usage_row.cached_input_tokens,
            )
            if (
                detail.get("price_version") == desired_version
                and Decimal(str(usage_row.cost_cny or 0)) == new_cost
            ):
                continue
            detail["price_version"] = desired_version
            usage_row.detail_json = detail
            usage_row.cost_cny = new_cost
            rows_updated += 1
            affected_run_ids.add(usage_row.run_id)
        for run_id in affected_run_ids:
            run_total = await session.scalar(
                select(func.coalesce(func.sum(RunUsage.cost_cny), 0)).where(
                    RunUsage.run_id == run_id, RunUsage.category == "llm"
                )
            )
            run_row = await session.get(Run, run_id)
            if run_row is not None:
                run_row.estimated_cost_cny = Decimal(str(run_total or 0))
        await session.commit()
    return {"rows_updated": rows_updated, "runs_updated": len(affected_run_ids)}


async def _main() -> None:
    config = get_service_config()
    engine = make_engine(config.database_url)
    try:
        summary = await recompute_llm_costs(make_session_factory(engine))
        print(f"cost_recompute: {summary}")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(_main())
