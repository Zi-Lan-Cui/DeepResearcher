"""Run control-plane admission service.

This service persists accepted work as ``queued``. It deliberately knows nothing
about asyncio tasks, LangGraph, checkpoints, or workers.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from sqlalchemy import func, select

from deepsearch_agent.observability.tracing.context import new_id
from deepsearch_agent.service.models import Run
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
        # M2 单进程内封住 count+insert；M3 用 DB 约束/事务取代。
        self._admission_lock = asyncio.Lock()

    async def create(self, user_id: int, query: str) -> str:
        query = query.strip()
        async with self._admission_lock:
            async with self._session_factory() as session:
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
