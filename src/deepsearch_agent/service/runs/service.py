"""Run control-plane admission service.

This service persists accepted work as ``queued``. It deliberately knows nothing
about asyncio tasks, LangGraph, checkpoints, or workers.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from sqlalchemy import func, select, text

from deepsearch_agent.observability.tracing.context import new_id
from deepsearch_agent.service.coordination import RUN_ADMISSION_LOCK_ID
from deepsearch_agent.service.persistence.models import Run
from deepsearch_agent.service.settings import ServiceConfig


class QuotaExceededError(Exception):
    """The user's accepted, unfinished run count reached its quota."""


class RunService:
    """Apply admission policy and persist durable queued work."""

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
