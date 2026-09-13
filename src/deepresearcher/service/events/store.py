"""Database-authoritative RunEvent sequence allocation and persistence."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from sqlalchemy import select, update

from deepresearcher.service.persistence.models import Run, RunEvent
from deepresearcher.service.signals import PostgresSignalBus


class RunEventStore:
    def __init__(
        self,
        session_factory: Callable[[], Any],
        *,
        publish_persisted: Callable[[str, Sequence[dict]], None] | None = None,
        signal_bus: PostgresSignalBus | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._publish_persisted = publish_persisted
        self._signal_bus = signal_bus

    async def append(self, run_id: str, records: Sequence[dict]) -> list[dict]:
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
        if self._publish_persisted is not None:
            self._publish_persisted(run_id, assigned)
        if self._signal_bus is not None:
            await self._signal_bus.notify_event(run_id)
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
