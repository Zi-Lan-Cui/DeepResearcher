"""保留期清理：终态 run 过期后清除过程数据，报告本体保存。"""

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from sqlalchemy import delete, select, update

from deepresearcher.service.coordination.keys import RUN_RETENTION_SWEEP_LOCK_ID
from deepresearcher.service.coordination.locks import try_acquire_xact_lock
from deepresearcher.service.storage.tables import TERMINAL_STATUSES, Run, RunEvent, RunUsage


def prune_event_jsonl(events_dir: Path, retention_days: int) -> int:
    """按 mtime 删除超龄的 run 事件 JSONL，返回删除数。

    清理是尽力而为：单个文件出错跳过，清理失败不阻塞进程启动。
    retention_days<=0 表示关闭清理。
    """
    if retention_days <= 0 or not events_dir.is_dir():
        return 0
    cutoff = time.time() - retention_days * 86_400
    removed = 0
    for path in events_dir.glob("run-*.jsonl"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError:
            continue
    return removed


async def sweep_run_traces(
    session_factory: Callable[[], Any],
    checkpointer: Any,
    *,
    retention_days: int,
    now: datetime | None = None,
    batch_size: int = 200,
) -> dict[str, int]:
    """清除超过保留期的终态 run 的过程数据并标记"""
    if retention_days <= 0:
        return {}
    reference = now or datetime.now(timezone.utc)
    cutoff = reference - timedelta(days=retention_days)
    async with session_factory() as session:
        async with session.begin():
            if not await try_acquire_xact_lock(session, RUN_RETENTION_SWEEP_LOCK_ID):
                return {}
            run_ids = list(
                await session.scalars(
                    select(Run.id)
                    .where(
                        Run.status.in_(TERMINAL_STATUSES),
                        Run.finished_at.is_not(None),
                        Run.finished_at < cutoff,
                        Run.retention_pruned_at.is_(None),
                    )
                    .order_by(Run.finished_at)
                    .limit(batch_size)
                )
            )
            if not run_ids:
                return {}
            events = await session.execute(delete(RunEvent).where(RunEvent.run_id.in_(run_ids)))
            usage = await session.execute(delete(RunUsage).where(RunUsage.run_id.in_(run_ids)))
            await session.execute(
                update(Run).where(Run.id.in_(run_ids)).values(retention_pruned_at=reference)
            )
    threads = 0
    if checkpointer is not None:
        for run_id in run_ids:
            await checkpointer.adelete_thread(run_id)
            threads += 1
    return {
        "runs": len(run_ids),
        "events": int(events.rowcount or 0),
        "usage": int(usage.rowcount or 0),
        "threads": threads,
    }
