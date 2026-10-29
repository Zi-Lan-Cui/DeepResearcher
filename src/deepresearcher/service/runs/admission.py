"""run 准入策略：配额检查通过后把受理的工作持久为 queued。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from deepresearcher.observability.span import new_id
from deepresearcher.service.coordination.keys import RUN_ADMISSION_LOCK_ID
from deepresearcher.service.coordination.locks import acquire_xact_lock
from deepresearcher.service.messaging.publisher import DispatchPublisher
from deepresearcher.service.runs.transitions import OUTSTANDING_STATUSES
from deepresearcher.service.settings import ServiceConfig
from deepresearcher.service.storage.tables import TERMINAL_STATUSES, Run


class QuotaExceededError(Exception):
    """该用户已受理且未完成的 run 数达到配额。"""


class RunAdmission:
    """执行准入策略并持久化排队工作。"""

    def __init__(
        self,
        *,
        session_factory: Callable[[], Any],
        config: ServiceConfig,
        publisher: DispatchPublisher,
    ) -> None:
        self._session_factory = session_factory
        self._config = config
        self._publisher = publisher

    async def _active_run_for_request(self, user_id: int, request_id: str) -> str | None:
        """同一幂等键下尚未终态的行；终态行不参与去重，允许重试。"""
        async with self._session_factory() as session:
            return await session.scalar(
                select(Run.id).where(
                    Run.user_id == user_id,
                    Run.request_id == request_id,
                    Run.status.not_in(TERMINAL_STATUSES),
                )
            )

    async def create(
        self,
        user_id: int,
        query: str,
        preferences_snapshot: dict[str, int] | None = None,
        *,
        request_id: str | None = None,
    ) -> str:
        """受理新研究问题：配额通过后写 queued 行，并保证它一定被分派。

        参数：
            user_id: 提交者。
            query: 已去除首尾空白的研究问题。
            preferences_snapshot: 创建时刻冻结的个人偏好。
            request_id: 客户端的幂等键；为 None 表示不去重。

        返回：
            str: 新 run 的 id；重复提交时返回既有 run 的 id。

        抛出：
            QuotaExceededError: 个人在途或全局排队已达上限。
        """
        query = query.strip()
        if request_id:
            # 快速路径：同一 request_id 已在途，直接返回，不发任何消息。
            existing = await self._active_run_for_request(user_id, request_id)
            if existing is not None:
                return existing
        run_id = new_id("run")
        frozen = dict(preferences_snapshot) if preferences_snapshot is not None else None

        async def local_transaction() -> None:
            """只做「计数 + 插入」，内部不发起网络调用。"""
            async with self._session_factory() as session:
                async with session.begin():
                    # 用户配额与全局 queued 上限都是跨行不变量。一把短事务锁把所有
                    # API 副本的「计数 + INSERT」串行化，commit/rollback 自动释放。
                    await acquire_xact_lock(session, RUN_ADMISSION_LOCK_ID)
                    outstanding = await session.scalar(
                        select(func.count())
                        .select_from(Run)
                        .where(
                            Run.user_id == user_id,
                            Run.status.in_(OUTSTANDING_STATUSES),
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
                            f"系统等待队列已满（{self._config.max_global_queued_runs}），"
                            "请稍后重试。"
                        )
                    session.add(
                        Run(
                            id=run_id,
                            user_id=user_id,
                            query=query,
                            request_id=request_id,
                            status="queued",
                            preferences_snapshot=frozen,
                        )
                    )

        try:
            await self._publisher.dispatch(run_id, local_transaction)
        except IntegrityError:
            # 并发同键：唯一索引挡住了第二条插入，本地事务已回滚，可以直接回查既有行。
            existing = (
                await self._active_run_for_request(user_id, request_id) if request_id else None
            )
            if existing is None:
                raise
            return existing
        return run_id
