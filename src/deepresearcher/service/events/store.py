"""以数据库为权威的 RunEvent 序号分配与持久化。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from sqlalchemy import select, update

from deepresearcher.service.persistence.models import Run, RunEvent


class RunEventStore:
    def __init__(self, session_factory: Callable[[], Any]) -> None:
        self._session_factory = session_factory

    async def append(self, run_id: str, records: Sequence[dict]) -> list[dict]:
        """事务内分配 seq 并写入数据库；返回带 seq 的完整记录。

        本方法只负责持久化，不做投递与通知；唯一允许的调用者是 RunEventHub.flush，
        它在 append 成功后发 notify。其他位置直调本方法时消费端收不到唤醒，
        只能等轮询追平。轮询是回退路径，正确性不丢，但这绕过了唯一投递点。
        """
        if not records:
            return []
        async with self._session_factory() as session:
            result = await session.execute(
                update(Run)
                .where(Run.id == run_id)
                .values(event_seq=Run.event_seq + len(records))
                .returning(Run.event_seq)
            )
            final_seq = result.scalar_one_or_none()
            if final_seq is None:
                return []
            first = int(final_seq) - len(records) + 1
            assigned = []
            for offset, record in enumerate(records):
                item = dict(record)
                item["run_id"] = run_id
                item["seq"] = first + offset
                assigned.append(item)
                session.add(
                    RunEvent(
                        run_id=run_id,
                        seq=item["seq"],
                        event_type=str(item.get("event_type", "")),
                        record=item,
                    )
                )
            await session.commit()
        return assigned

    async def after(self, run_id: str, seq: int) -> list[RunEvent]:
        async with self._session_factory() as session:
            return list(
                (
                    await session.scalars(
                        select(RunEvent)
                        .where(RunEvent.run_id == run_id, RunEvent.seq > seq)
                        .order_by(RunEvent.seq)
                    )
                ).all()
            )
