"""按日 token 用量统计：账户页图表的数据源。"""

from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, Request
from sqlalchemy import func, select

from deepresearcher.service.storage.tables import Run, User
from deepresearcher.service.web.dependencies import app_state, current_user

router = APIRouter(prefix="/api/stats")

STATS_WINDOW_DAYS = (7, 30, 90)


@router.get("/daily")
async def stats_daily(
    request: Request,
    user: User = Depends(current_user),
    days: int = 30,
) -> dict[str, Any]:
    """最近 N 天逐日聚合：runs 数、输入/输出/已缓存 token 与输入侧命中率。

    日期按 UTC 分组；无 run 的日期不补零，缺位由前端图表处理。
    """
    if days not in STATS_WINDOW_DAYS:
        days = 30
    cutoff = datetime.now(UTC) - timedelta(days=days)
    async with app_state(request).session_factory() as session:
        rows = (
            await session.execute(
                select(
                    func.date(Run.created_at),
                    func.count(),
                    func.coalesce(func.sum(Run.input_tokens), 0),
                    func.coalesce(func.sum(Run.output_tokens), 0),
                    func.coalesce(func.sum(Run.cached_input_tokens), 0),
                )
                .where(Run.user_id == user.id, Run.created_at >= cutoff)
                .group_by(func.date(Run.created_at))
                .order_by(func.date(Run.created_at))
            )
        ).all()
    series = []
    for day, runs, input_tokens, output_tokens, cached_input in rows:
        series.append(
            {
                "date": str(day),
                "runs": int(runs),
                "input_tokens": int(input_tokens),
                "output_tokens": int(output_tokens),
                "cached_input_tokens": int(cached_input),
                "cache_hit_rate": round(int(cached_input) / int(input_tokens), 4)
                if int(input_tokens) > 0
                else None,
            }
        )
    return {"days": days, "series": series}
