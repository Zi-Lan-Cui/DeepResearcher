"""以数据库为权威的 RunEvent 序号分配与持久化。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from deepresearcher.service.storage.tables import Run, RunEvent


class RunEventStore:
    def __init__(self, session_factory: Callable[[], Any]) -> None:
        self._session_factory = session_factory

    async def append(self, run_id: str, records: Sequence[dict]) -> list[dict]:
        """事务内分配 seq 并写入数据库；返回带 seq 的完整记录。

        写入前按 dedup_key 过滤：批内同键留首条，库中已有的整条丢弃；无键记录一律
        写入。seq 只为实际写入的条数分配，去重不留下空洞。
        """
        if not records:
            return []
        batch = _dedup_within_batch(records)
        try:
            return await self._write_new_records(run_id, batch)
        except IntegrityError:
            # 只有并发写入会到这里：预查时对方尚未提交，唯一约束要等 commit 才拦下。
            # 事务已整体回滚，重跑一遍预查即可看到对方那几行。
            return await self._write_new_records(run_id, batch)

    async def _write_new_records(self, run_id: str, records: Sequence[dict]) -> list[dict]:
        async with self._session_factory() as session:
            to_insert = await self._drop_already_stored(session, run_id, records)
            if not to_insert:
                return []
            result = await session.execute(
                update(Run)
                .where(Run.id == run_id)
                .values(event_seq=Run.event_seq + len(to_insert))
                .returning(Run.event_seq)
            )
            final_seq = result.scalar_one_or_none()
            if final_seq is None:
                return []
            first_seq = int(final_seq) - len(to_insert) + 1
            written_records = []
            for index, record in enumerate(to_insert):
                new_record = dict(record)
                new_record["run_id"] = run_id
                new_record["seq"] = first_seq + index
                written_records.append(new_record)
                session.add(
                    RunEvent(
                        run_id=run_id,
                        seq=new_record["seq"],
                        event_type=str(new_record.get("event_type", "")),
                        dedup_key=_dedup_key_or_empty(new_record) or None,
                        record=new_record,
                    )
                )
            await session.commit()
        return written_records

    async def _drop_already_stored(
        self, session: Any, run_id: str, records: Sequence[dict]
    ) -> list[dict]:
        """返回库里还没有的那些记录，保持原顺序。"""
        batch_keys = [key for record in records if (key := _dedup_key_or_empty(record))]
        if not batch_keys:
            return list(records)
        stored_keys = set(
            (
                await session.scalars(
                    select(RunEvent.dedup_key).where(
                        RunEvent.run_id == run_id, RunEvent.dedup_key.in_(batch_keys)
                    )
                )
            ).all()
        )
        return [record for record in records if _dedup_key_or_empty(record) not in stored_keys]

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


def _dedup_key_or_empty(record: dict) -> str:
    """取记录的去重键；缺失或非字符串时返回空串，调用方据此判断是否参与去重。"""
    value = record.get("dedup_key")
    return value if isinstance(value, str) else ""


def _dedup_within_batch(records: Sequence[dict]) -> list[dict]:
    """批内同键只留首条，保持原有顺序；无键记录一律保留。"""
    seen_keys: set[str] = set()
    kept: list[dict] = []
    for record in records:
        key = _dedup_key_or_empty(record)
        if key:
            if key in seen_keys:
                continue
            seen_keys.add(key)
        kept.append(dict(record))
    return kept
