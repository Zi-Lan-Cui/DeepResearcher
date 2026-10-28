"""保留期清扫:sweep_run_traces 的候选谓词、幂等打标、分批与开关。"""

from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy import func, select

from deepresearcher.service.runs.retention import sweep_run_traces
from deepresearcher.service.storage.engine import make_engine, make_session_factory
from deepresearcher.service.storage.tables import Run, RunEvent, RunUsage, User
from pg_backend import database_url_for

pytestmark = pytest.mark.asyncio

NOW = datetime(2026, 10, 12, tzinfo=timezone.utc)


class FakeCheckpointer:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    async def adelete_thread(self, thread_id: str) -> None:
        self.deleted.append(thread_id)


async def _seed(session_factory) -> None:
    async with session_factory() as session:
        session.add(User(id=1, email="r@test.dev", password_hash="h"))
        # 三场终态超窗(09-01)、一场终态未超窗(10-10)、一场非终态超窗(仍 running)
        old_ids = ["run-old-1", "run-old-2", "run-old-3"]
        for run_id in old_ids:
            session.add(
                Run(
                    id=run_id,
                    user_id=1,
                    query="旧",
                    status="completed",
                    finished_at=NOW - timedelta(days=30),
                    report_markdown="# 报告",
                )
            )
            session.add(RunEvent(run_id=run_id, seq=1, event_type="run_status", record={}))
            session.add(
                RunUsage(
                    id=f"usage-{run_id}",
                    run_id=run_id,
                    category="llm",
                    component="Supervisor",
                    status="ok",
                )
            )
        session.add(
            Run(
                id="run-recent",
                user_id=1,
                query="新",
                status="completed",
                finished_at=NOW - timedelta(days=2),
            )
        )
        session.add(Run(id="run-live", user_id=1, query="活", status="running"))
        await session.commit()


@pytest_asyncio.fixture
async def session_factory(tmp_path):
    engine = make_engine(database_url_for(tmp_path))
    factory = make_session_factory(engine)
    await _seed(factory)
    yield factory
    await engine.dispose()


async def test_sweep_prunes_expired_terminal_runs_and_keeps_reports(session_factory):
    checkpointer = FakeCheckpointer()
    summary = await sweep_run_traces(session_factory, checkpointer, retention_days=14, now=NOW)
    assert summary == {"runs": 3, "events": 3, "usage": 3, "threads": 3}
    assert sorted(checkpointer.deleted) == ["run-old-1", "run-old-2", "run-old-3"]

    async with session_factory() as session:
        # runs 行与报告本体永存，只打标
        pruned = await session.scalars(select(Run.id).where(Run.retention_pruned_at.is_not(None)))
        assert sorted(pruned) == ["run-old-1", "run-old-2", "run-old-3"]
        report = await session.scalar(select(Run.report_markdown).where(Run.id == "run-old-1"))
        assert report == "# 报告"
        # 未超窗与非终态的留档原样在
        kept_events = await session.scalar(select(func.count()).select_from(RunEvent))
        assert kept_events == 0  # 仅旧三场有事件；recent/live 未播种事件
        assert await session.get(Run, "run-recent") is not None
        assert await session.get(Run, "run-live") is not None
        remaining_usage = await session.scalar(select(func.count()).select_from(RunUsage))
        assert remaining_usage == 0


async def test_sweep_is_idempotent_and_batched(session_factory):
    checkpointer = FakeCheckpointer()
    first = await sweep_run_traces(
        session_factory, checkpointer, retention_days=14, now=NOW, batch_size=2
    )
    assert first["runs"] == 2
    second = await sweep_run_traces(
        session_factory, checkpointer, retention_days=14, now=NOW, batch_size=2
    )
    assert second["runs"] == 1  # 批限分轮消化
    third = await sweep_run_traces(session_factory, checkpointer, retention_days=14, now=NOW)
    assert third == {}  # 打标后不入候选，重扫无需再次判定
    assert len(checkpointer.deleted) == 3


async def test_sweep_disabled_or_without_window(session_factory):
    assert await sweep_run_traces(session_factory, None, retention_days=0, now=NOW) == {}
    # 窗口大到没人超窗：无事发生，checkpointer 为 None 也不触碰
    assert await sweep_run_traces(session_factory, None, retention_days=3650, now=NOW) == {}
