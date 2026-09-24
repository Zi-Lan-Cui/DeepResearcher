"""run 控制面准入服务。

本服务把受理的工作持久为 ``queued``。它刻意不知道 asyncio 任务、
LangGraph、checkpoint 或 worker。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from sqlalchemy import func, select, text

from deepresearcher.observability.tracing.context import new_id
from deepresearcher.service.coordination import RUN_ADMISSION_LOCK_ID
from deepresearcher.service.persistence.models import Run
from deepresearcher.service.settings import ServiceConfig


class QuotaExceededError(Exception):
    """该用户已受理且未完成的 run 数达到配额。"""


class RunService:
    """执行准入策略并持久化排队工作。"""

    def __init__(
        self,
        *,
        session_factory: Callable[[], Any],
        config: ServiceConfig,
    ) -> None:
        self._session_factory = session_factory
        self._config = config
        # SQLite 测试/单进程兼容路径由本地锁保护；PostgreSQL 在
        # create() 事务内再取全局 advisory lock，协调多 API 进程。
        self._admission_lock = asyncio.Lock()

    async def create(self, user_id: int, query: str) -> str:
        query = query.strip()
        async with self._admission_lock:
            async with self._session_factory() as session:
                bind = session.get_bind()
                if bind.dialect.name == "postgresql":
                    # 用户配额和全局 queued 上限都是跨行不变量。一把短事务锁
                    # 将所有 API 副本的“计数 + INSERT”串行化，commit/rollback 自动释放。
                    await session.execute(
                        text("SELECT pg_advisory_xact_lock(:lock_id)"),
                        {"lock_id": RUN_ADMISSION_LOCK_ID},
                    )
                outstanding = await session.scalar(
                    select(func.count())
                    .select_from(Run)
                    .where(
                        Run.user_id == user_id,
                        Run.status.in_(("queued", "running", "interrupted")),
                    )
                )
                if (outstanding or 0) >= self._config.max_concurrent_runs_per_user:
                    raise QuotaExceededError(
                        "同时进行或排队的运行已达上限"
                        f"（{self._config.max_concurrent_runs_per_user}）。"
                    )
                queued = await session.scalar(
                    select(func.count()).select_from(Run).where(Run.status == "queued")
                )
                if (queued or 0) >= self._config.max_global_queued_runs:
                    raise QuotaExceededError(
                        f"系统等待队列已满（{self._config.max_global_queued_runs}），请稍后重试。"
                    )
                run_id = new_id("run")
                session.add(Run(id=run_id, user_id=user_id, query=query, status="queued"))
                await session.commit()
        return run_id
