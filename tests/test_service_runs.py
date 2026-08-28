import asyncio
from types import SimpleNamespace

import pytest
import pytest_asyncio
from sqlalchemy import select

from deepsearch_agent.config import (
    AgentConfig,
    AppConfig,
    LLMConfig,
    ObservabilityConfig,
    SearchConfig,
    Settings,
)
from deepsearch_agent.service.db import init_db, make_engine, make_session_factory
from deepsearch_agent.service.events import FanoutSink
from deepsearch_agent.service.models import Run, RunEvent
from deepsearch_agent.service.runs import QuotaExceededError, RunManager
from deepsearch_agent.service.settings import ServiceConfig

pytestmark = pytest.mark.asyncio

USER_ID = 1


def _settings(tmp_path) -> Settings:
    return Settings(
        llm=LLMConfig(),
        agent=AgentConfig(),
        search=SearchConfig(tavily_api_key="test"),
        app=AppConfig(),
        observability=ObservabilityConfig(log_dir=tmp_path),
    )


class FakeGraph:
    """记录 ainvoke 输入；可选先发一条引擎事件、等待门控、返回结果或抛错。"""

    def __init__(self, *, result=None, error=None, gate=None, emit_events=1):
        self.result = result or {}
        self.error = error
        self.gate = gate
        self.emit_events = emit_events
        self.ainvoke_inputs: list[dict] = []

    async def ainvoke(self, input, **_kwargs):  # noqa: A002 - 与 LangGraph 契约同名
        self.ainvoke_inputs.append(dict(input))
        for i in range(self.emit_events):
            self._sink.write(
                {"run_id": input["run_id"], "event_type": f"engine_{i}", "payload": {}}
            )
        if self.gate is not None:
            await self.gate.wait()
        if self.error is not None:
            raise self.error
        return self.result


def _completed_result():
    return {
        "run": SimpleNamespace(
            phase="completed", terminal_reason="report_rendered", error=None
        ),
        "answer_mode": "deep_research",
        "report": "# 研究报告\n完成。",
        "citations": [{"id": "e1", "url": "https://a", "title": "A", "quote": "q", "claim": "c"}],
        "evidence_count": 41,
        "source_count": 12,
    }


@pytest_asyncio.fixture
async def manager(tmp_path):
    # 用文件库而非 :memory:+StaticPool：StaticPool 全共享一条连接，后台任务与
    # cancel() 的并发 session 会互相踩（一方回滚会 terminate 另一方在用的连接）。
    # 生产 asyncpg 池每 session 独立连接，无此问题——文件 SQLite 还原该语义。
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path / 'runs.db'}")
    await init_db(engine)
    session_factory = make_session_factory(engine)
    async with session_factory() as session:
        from deepsearch_agent.service.models import User

        session.add(User(id=USER_ID, email="u@test", password_hash="h"))
        await session.commit()
    fanout = FanoutSink(asyncio.get_running_loop())
    config = ServiceConfig(
        database_url="unused",
        jwt_secret="s" * 40,
        service_log_dir=tmp_path,
        jsonl_events=False,
        max_concurrent_runs_per_user=2,
    )
    holder: dict = {}

    def graph_factory(*, settings, event_sink, http_client):
        holder["sink"] = event_sink
        graph = holder.get("graph") or FakeGraph()
        graph._sink = event_sink
        return graph

    manager = RunManager(
        settings=_settings(tmp_path),
        session_factory=session_factory,
        config=config,
        fanout=fanout,
        http_client=SimpleNamespace(),
        graph_factory=graph_factory,
    )
    manager.holder = holder  # type: ignore[attr-defined]
    manager.engine = engine  # type: ignore[attr-defined]
    manager.session_factory = session_factory  # type: ignore[attr-defined]
    manager.fanout = fanout  # type: ignore[attr-defined]
    yield manager
    # 先收敛所有后台任务（它们的 finally 还要写库），再拆引擎，避免
    # “closed database / no such table” 竞态。
    await manager.shutdown()
    await engine.dispose()


async def _settle(manager, run_id):
    task = manager._tasks.get(run_id)  # noqa: SLF001 - 测试观察内部收尾
    if task is not None:
        await task


async def _row(manager, run_id):
    async with manager.session_factory() as session:
        return await session.get(Run, run_id)


async def test_success_persists_terminal_and_injects_run_id(manager):
    graph = FakeGraph(result=_completed_result())
    manager.holder["graph"] = graph
    run_id = await manager.start(USER_ID, "  测试问题  ")
    await _settle(manager, run_id)

    # 不变式 1：run_id/session_id 必须显式进入引擎输入
    assert graph.ainvoke_inputs[0] == {"query": "测试问题", "run_id": run_id, "session_id": run_id}
    run = await _row(manager, run_id)
    assert run.status == "completed"
    assert run.terminal_reason == "report_rendered"
    assert run.report_markdown.startswith("# 研究报告")
    assert run.citations_json == graph.result["citations"]
    assert (run.evidence_count, run.source_count) == (41, 12)
    assert run.query == "测试问题"  # strip 生效


async def test_failure_persists_failed_with_truncated_message(manager):
    manager.holder["graph"] = FakeGraph(error=RuntimeError("x" * 900))
    run_id = await manager.start(USER_ID, "q")
    await _settle(manager, run_id)
    run = await _row(manager, run_id)
    assert run.status == "failed"
    assert run.terminal_reason == "run_exception"
    assert len(run.error_message) <= 500


async def test_cancel_mid_run_persists_cancelled_once(manager):
    gate = asyncio.Event()
    manager.holder["graph"] = FakeGraph(gate=gate, result=_completed_result())
    run_id = await manager.start(USER_ID, "q")
    while run_id not in manager._tasks or manager._tasks[run_id].done():  # noqa: SLF001
        await asyncio.sleep(0.01)
    await manager.cancel(USER_ID, run_id)
    gate.set()
    await _settle(manager, run_id)

    run = await _row(manager, run_id)
    assert run.status == "cancelled"
    assert run.terminal_reason == "user_cancelled"
    assert run.report_markdown is None  # 迟到的成功结果不得覆盖已写终态

    async with manager.session_factory() as session:
        dones = (
            await session.scalars(
                select(RunEvent).where(
                    RunEvent.run_id == run_id, RunEvent.event_type == "run_done"
                )
            )
        ).all()
    assert len(dones) == 1  # 不变式 2/3：done 恰一次且在 DB（可回放）


async def test_cancel_after_completion_is_idempotent(manager):
    manager.holder["graph"] = FakeGraph(result=_completed_result())
    run_id = await manager.start(USER_ID, "q")
    await _settle(manager, run_id)
    again = await manager.cancel(USER_ID, run_id)
    assert again.status == "completed"


async def test_cancel_other_users_run_raises_lookup(manager):
    run_id = await manager.start(USER_ID, "q")
    with pytest.raises(LookupError):
        await manager.cancel(999, run_id)
    await manager.cancel(USER_ID, run_id)  # 清理：让 fixture 无悬挂任务
    await _settle(manager, run_id)


async def test_quota_blocks_third_concurrent_run(manager):
    gate = asyncio.Event()
    manager.holder["graph"] = FakeGraph(gate=gate, result=_completed_result())
    first = await manager.start(USER_ID, "q1")
    second = await manager.start(USER_ID, "q2")
    with pytest.raises(QuotaExceededError):
        await manager.start(USER_ID, "q3")
    gate.set()
    for run_id in (first, second):
        await _settle(manager, run_id)


async def test_events_are_persisted_in_seq_order_with_done_last(manager):
    graph = FakeGraph(result=_completed_result(), emit_events=3)
    manager.holder["graph"] = graph
    run_id = await manager.start(USER_ID, "q")
    await _settle(manager, run_id)

    async with manager.session_factory() as session:
        events = (
            await session.scalars(
                select(RunEvent).where(RunEvent.run_id == run_id).order_by(RunEvent.seq)
            )
        ).all()
    types = [event.event_type for event in events]
    seqs = [event.seq for event in events]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)  # 无洞、无重复
    assert types[0] == "run_status"      # queued 最早
    assert types[-1] == "run_done"       # done 严格最后（重连回放以此收尾）
    assert "engine_0" in types and "engine_2" in types
    assert types.count("run_done") == 1


async def test_reconcile_startup_converts_stale_rows(manager):
    async with manager.session_factory() as session:
        session.add_all(
            [
                Run(id="run-stale-a", user_id=USER_ID, query="q", status="running"),
                Run(id="run-stale-b", user_id=USER_ID, query="q", status="queued"),
                Run(id="run-done", user_id=USER_ID, query="q", status="completed"),
            ]
        )
        await session.commit()
    changed = await manager.reconcile_startup()
    assert changed == 2
    for run_id, expected in (("run-stale-a", "failed"), ("run-stale-b", "failed"), ("run-done", "completed")):
        run = await _row(manager, run_id)
        assert run.status == expected
    assert (await _row(manager, "run-stale-a")).terminal_reason == "server_restart"
    # 孤儿 run 的事件流必须有合成 done 收尾，否则历史详情页 SSE 无限重连。
    async with manager.session_factory() as session:
        orphans = (
            await session.scalars(
                select(RunEvent).where(
                    RunEvent.run_id.in_(("run-stale-a", "run-stale-b")),
                    RunEvent.event_type == "run_done",
                )
            )
        ).all()
        completed_done = (
            await session.scalars(
                select(RunEvent).where(
                    RunEvent.run_id == "run-done", RunEvent.event_type == "run_done"
                )
            )
        ).all()
    assert len(orphans) == 2
    assert all(event.record["payload"]["status"] == "failed" for event in orphans)
    assert completed_done == []  # 非孤儿不补
