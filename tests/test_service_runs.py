import asyncio
import traceback
from dataclasses import replace
from datetime import datetime, timedelta, timezone
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
from deepsearch_agent.service.events.store import RunEventStore
from deepsearch_agent.service.events.stream import CLOSE_STREAM as CLOSE_STREAM_FLAG
from deepsearch_agent.service.events.stream import FanoutSink
from deepsearch_agent.service.persistence.database import init_db, make_engine, make_session_factory
from deepsearch_agent.service.persistence.models import Run, RunEvent, User
from deepsearch_agent.service.runs.manager import RunManager
from deepsearch_agent.service.runs.queue import PostgresRunQueue
from deepsearch_agent.service.runs.service import QuotaExceededError, RunService
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

    def __init__(
        self,
        *,
        result=None,
        error=None,
        gate=None,
        emit_events=1,
        stream_messages=(),
        resume_run_id="run-resume",
        interrupt_payload=None,
    ):
        self.result = result or {}
        self.error = error
        self.gate = gate
        self.emit_events = emit_events
        self.resume_run_id = resume_run_id
        self.stream_messages = list(stream_messages)  # [(namespace_tuple, text)]
        self.ainvoke_inputs: list[dict] = []
        self.seen_configs: list[dict] = []
        self.none_inputs = 0
        self.interrupt_payload = interrupt_payload

    async def _run(self, input):  # noqa: A002 - 与 LangGraph 契约同名
        run_id = input["run_id"] if input is not None else self.resume_run_id
        if input is None:  # resume 形态：astream(None)，thread_id 定位断点
            self.none_inputs += 1
        else:
            self.ainvoke_inputs.append(dict(input))
        for i in range(self.emit_events):
            self._sink.write({"run_id": run_id, "event_type": f"engine_{i}", "payload": {}})
        if self.gate is not None:
            await self.gate.wait()
        if self.error is not None:
            raise self.error
        return self.result

    async def ainvoke(self, input, **_kwargs):
        return await self._run(input)

    async def astream(self, input, **kwargs):
        """复刻官方形态：subgraphs=True 时 yield (namespace, mode, chunk)。

        stream_messages 条目 = (namespace, text) 或 (namespace, text, msg_type)，
        msg_type 默认 "ai"；"tool" 模拟 messages 模式里混入的工具回执。
        """
        self.seen_configs.append(dict(kwargs))  # 记录 config（thread_id 注入断言用）
        for entry in self.stream_messages:
            namespace, text = entry[0], entry[1]
            msg_type = entry[2] if len(entry) > 2 else "ai"
            chunk = (
                SimpleNamespace(type=msg_type, content_blocks=[{"type": "text", "text": text}]),
                {},
            )
            yield (namespace, "messages", chunk)
        if self.interrupt_payload is not None:
            yield (
                (),
                "updates",
                {"__interrupt__": (SimpleNamespace(value=self.interrupt_payload),)},
            )
            return
        yield ((), "values", await self._run(input))


def _completed_result():
    return {
        "run": SimpleNamespace(phase="completed", terminal_reason="report_rendered", error=None),
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
        from deepsearch_agent.service.persistence.models import User

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

    def graph_factory(*, settings, event_sink, http_client, checkpointer=None, tool_cache=None):
        del tool_cache
        holder["sink"] = event_sink
        holder["checkpointer"] = checkpointer
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
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=5)
        except asyncio.CancelledError:
            return
        except TimeoutError:
            stacks = "".join(
                traceback.format_list(traceback.extract_stack(frame)) for frame in task.get_stack()
            )
            pytest.fail(f"run task did not settle:\n{stacks}")


async def _row(manager, run_id):
    async with manager.session_factory() as session:
        return await session.get(Run, run_id)


async def test_manager_delegates_execution_without_building_graph_itself(manager):
    captured = {}
    original_execute = manager._executor.execute  # noqa: SLF001

    async def execute(run_id, user_id, query, **kwargs):
        captured.update(
            run_id=run_id,
            user_id=user_id,
            query=query,
            kwargs=kwargs,
        )
        await original_execute(run_id, user_id, query, **kwargs)

    manager._executor.execute = execute  # noqa: SLF001 - 验证 M1 职责接缝
    run_id = await manager.start(USER_ID, "  委托执行  ")
    await _settle(manager, run_id)

    assert captured["run_id"] == run_id
    assert captured["user_id"] == USER_ID
    assert captured["query"] == "委托执行"
    assert captured["kwargs"]["resume"] is False
    assert captured["kwargs"]["resume_input"] is None
    assert captured["kwargs"]["claim"].run_id == run_id
    assert captured["kwargs"]["claim"].claimed is True


async def test_running_status_is_announced_only_for_user_visible_claim(manager):
    """Human resume leaves queued only when waiting; restart resume keeps resuming."""

    silent_id = "run-system-resume"
    announced_id = "run-human-resume"
    async with manager.session_factory() as session:
        session.add_all(
            [
                Run(id=silent_id, user_id=USER_ID, query="system", status="interrupted"),
                Run(id=announced_id, user_id=USER_ID, query="human", status="queued"),
            ]
        )
        await session.commit()
    manager.fanout.open(silent_id)
    manager.fanout.open(announced_id)

    assert await manager._executor._mark_running(  # noqa: SLF001
        silent_id,
        announce_running=False,
    )
    assert manager.fanout.take_pending(silent_id) == []

    assert await manager._executor._mark_running(  # noqa: SLF001
        announced_id,
        announce_running=True,
    )
    assert manager.fanout.take_pending(announced_id) == [
        {
            "run_id": announced_id,
            "event_type": "run_status",
            "payload": {"status": "running"},
        }
    ]


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
    while (await _row(manager, run_id)).status != "running":
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
                select(RunEvent).where(RunEvent.run_id == run_id, RunEvent.event_type == "run_done")
            )
        ).all()
    assert len(dones) == 1  # 不变式 2/3：done 恰一次且在 DB（可回放）


async def test_remote_manager_cancel_is_observed_through_durable_intent(manager, tmp_path):
    gate = asyncio.Event()
    manager._worker._heartbeat_seconds = 0.01  # noqa: SLF001 - cancellation probe seam
    manager.holder["graph"] = FakeGraph(gate=gate, result=_completed_result())
    run_id = await manager.start(USER_ID, "remote cancel")
    while (await _row(manager, run_id)).status != "running":
        await asyncio.sleep(0.01)

    controller = RunManager(
        settings=_settings(tmp_path),
        session_factory=manager.session_factory,
        config=manager._config,
        fanout=FanoutSink(asyncio.get_running_loop()),
        http_client=SimpleNamespace(),
        graph_factory=lambda **_kwargs: FakeGraph(),
    )
    requested = await controller.cancel(USER_ID, run_id)
    assert requested.status == "running"
    assert requested.cancellation_requested_at is not None
    await _settle(manager, run_id)
    assert (await _row(manager, run_id)).status == "cancelled"
    await controller.shutdown()


async def test_cancel_after_completion_is_idempotent(manager):
    manager.holder["graph"] = FakeGraph(result=_completed_result())
    run_id = await manager.start(USER_ID, "q")
    await _settle(manager, run_id)
    again = await manager.cancel(USER_ID, run_id)
    assert again.status == "completed"


async def test_shutdown_marks_interrupted_without_terminal_done(manager):
    gate = asyncio.Event()
    manager.holder["graph"] = FakeGraph(gate=gate, result=_completed_result())
    run_id = await manager.start(USER_ID, "q")
    while run_id not in manager._tasks or manager._tasks[run_id].done():  # noqa: SLF001
        await asyncio.sleep(0.01)

    await manager.shutdown()

    run = await _row(manager, run_id)
    assert run.status == "interrupted"
    assert run.terminal_reason == "server_shutdown"
    assert run.finished_at is None
    async with manager.session_factory() as session:
        dones = (
            await session.scalars(
                select(RunEvent).where(RunEvent.run_id == run_id, RunEvent.event_type == "run_done")
            )
        ).all()
    assert dones == []


async def test_graph_interrupt_persists_awaiting_input_without_done(manager):
    manager.holder["graph"] = FakeGraph(
        interrupt_payload={
            "kind": "clarification",
            "question": "你更关心哪一方面？",
            "options": ["成本", "效果", "风险"],
        }
    )
    run_id = await manager.start(USER_ID, "q")
    await _settle(manager, run_id)

    run = await _row(manager, run_id)
    assert run.status == "awaiting_input"
    assert run.finished_at is None
    async with manager.session_factory() as session:
        events = (
            await session.scalars(
                select(RunEvent).where(RunEvent.run_id == run_id).order_by(RunEvent.seq)
            )
        ).all()
    assert "clarification_requested" in [event.event_type for event in events]
    assert "run_done" not in [event.event_type for event in events]


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


async def test_global_queue_quota_rejects_new_admission(manager):
    service = RunService(
        session_factory=manager.session_factory,
        config=ServiceConfig(
            database_url="unused",
            jwt_secret="s" * 40,
            max_concurrent_runs_per_user=10,
            max_global_queued_runs=1,
        ),
    )
    await service.create(USER_ID, "q1")
    with pytest.raises(QuotaExceededError, match="等待队列已满"):
        await service.create(USER_ID, "q2")


async def test_two_queue_instances_claim_a_run_only_once(manager):
    service = RunService(session_factory=manager.session_factory, config=manager._config)
    run_id = await service.create(USER_ID, "atomic claim")
    first_queue = PostgresRunQueue(manager.session_factory)
    second_queue = PostgresRunQueue(manager.session_factory)

    claims = await asyncio.gather(
        first_queue.claim(worker_id="worker-a", lease_seconds=60),
        second_queue.claim(worker_id="worker-b", lease_seconds=60),
    )

    winners = [claim for claim in claims if claim is not None]
    assert len(winners) == 1
    assert winners[0].run_id == run_id
    assert winners[0].attempt == 1


async def test_worker_polling_claims_run_created_by_another_process(manager):
    """Independent Worker does not rely on the API process calling wake()."""
    manager._worker._poll_seconds = 0.01  # noqa: SLF001 - polling seam
    graph = FakeGraph(result=_completed_result(), emit_events=1)
    manager.holder["graph"] = graph
    service = RunService(session_factory=manager.session_factory, config=manager._config)
    run_id = await service.create(USER_ID, "external admission")

    await manager.start_worker()
    async with asyncio.timeout(2):
        while (await _row(manager, run_id)).status != "completed":
            await asyncio.sleep(0.01)
    await _settle(manager, run_id)

    assert len(graph.ainvoke_inputs) == 1
    async with manager.session_factory() as session:
        event_types = list(
            (
                await session.scalars(
                    select(RunEvent.event_type)
                    .where(RunEvent.run_id == run_id)
                    .order_by(RunEvent.seq)
                )
            ).all()
        )
    assert "engine_0" in event_types


async def test_stale_owner_cannot_renew_or_write_terminal_state(manager):
    service = RunService(session_factory=manager.session_factory, config=manager._config)
    run_id = await service.create(USER_ID, "owner CAS")
    queue = PostgresRunQueue(manager.session_factory)
    claim = await queue.claim(worker_id="worker-a", lease_seconds=60)
    assert claim is not None
    stale = replace(claim, lease_owner="worker-stale")

    assert await queue.renew(stale, lease_seconds=60) is False
    assert (
        await manager._executor.persist_status(  # noqa: SLF001 - owner CAS contract
            run_id, status="completed", claim=stale
        )
        is False
    )
    assert (await _row(manager, run_id)).status == "running"
    assert (
        await manager._executor.persist_status(  # noqa: SLF001 - owner CAS contract
            run_id, status="completed", claim=claim
        )
        is True
    )


async def test_expired_lease_is_reaped_for_resume(manager):
    service = RunService(session_factory=manager.session_factory, config=manager._config)
    run_id = await service.create(USER_ID, "expired")
    queue = PostgresRunQueue(manager.session_factory)
    claim = await queue.claim(worker_id="worker-a", lease_seconds=60)
    assert claim is not None
    async with manager.session_factory() as session:
        run = await session.get(Run, run_id)
        run.lease_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        await session.commit()

    reaped = await queue.reap_expired()
    assert [work.run_id for work in reaped] == [run_id]
    run = await _row(manager, run_id)
    assert run.status == "interrupted"
    assert run.terminal_reason == "lease_expired"
    assert run.lease_owner is None


async def test_two_event_stores_allocate_non_overlapping_sequences(manager):
    service = RunService(session_factory=manager.session_factory, config=manager._config)
    run_id = await service.create(USER_ID, "event sequence")
    first_store = RunEventStore(manager.session_factory)
    second_store = RunEventStore(manager.session_factory)

    batches = await asyncio.gather(
        first_store.append(run_id, [{"event_type": "from-a", "payload": {}}]),
        second_store.append(run_id, [{"event_type": "from-b", "payload": {}}]),
    )

    assert sorted(batch[0]["seq"] for batch in batches) == [1, 2]
    async with manager.session_factory() as session:
        seqs = list(
            (
                await session.scalars(
                    select(RunEvent.seq).where(RunEvent.run_id == run_id).order_by(RunEvent.seq)
                )
            ).all()
        )
    assert seqs == [1, 2]


async def test_global_capacity_keeps_excess_run_queued_then_dispatches(manager):
    gate = asyncio.Event()
    manager._worker._max_running = 1  # noqa: SLF001 - M2 容量接缝
    manager.holder["graph"] = FakeGraph(gate=gate, result=_completed_result())

    first = await manager.start(USER_ID, "q1")
    second = await manager.start(USER_ID, "q2")

    while (await _row(manager, first)).status != "running":
        await asyncio.sleep(0.01)
    assert (await _row(manager, second)).status == "queued"
    assert second not in manager._tasks  # noqa: SLF001

    gate.set()
    await _settle(manager, first)
    for _ in range(100):
        if (await _row(manager, second)).status != "queued":
            break
        await asyncio.sleep(0.01)
    await _settle(manager, second)
    assert (await _row(manager, second)).status == "completed"


async def test_cancel_queued_run_never_executes_graph(manager):
    gate = asyncio.Event()
    manager._worker._max_running = 1  # noqa: SLF001 - M2 容量接缝
    graph = FakeGraph(gate=gate, result=_completed_result())
    manager.holder["graph"] = graph

    first = await manager.start(USER_ID, "q1")
    queued = await manager.start(USER_ID, "q2")
    assert (await _row(manager, queued)).status == "queued"

    cancelled = await manager.cancel(USER_ID, queued)
    assert cancelled.status == "cancelled"
    assert queued not in manager._tasks  # noqa: SLF001

    gate.set()
    await _settle(manager, first)
    assert (await _row(manager, queued)).status == "cancelled"


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
    assert types[0] == "run_status"  # queued 最早
    assert types[-1] == "run_done"  # done 严格最后（重连回放以此收尾）
    assert "engine_0" in types and "engine_2" in types
    assert types.count("run_done") == 1


async def test_reconcile_startup_converts_stale_rows(manager):
    async with manager.session_factory() as session:
        session.add_all(
            [
                Run(id="run-stale-a", user_id=USER_ID, query="q", status="running"),
                Run(id="run-stale-b", user_id=USER_ID, query="q", status="queued"),
                Run(id="run-stale-c", user_id=USER_ID, query="q", status="interrupted"),
                Run(id="run-done", user_id=USER_ID, query="q", status="completed"),
            ]
        )
        await session.commit()
    killed, resumable = await manager.reconcile_startup()
    assert (killed, resumable) == (2, [])
    for run_id, expected in (
        ("run-stale-a", "failed"),
        ("run-stale-b", "queued"),  # 未领取任务在重启后仍可从头执行
        ("run-stale-c", "failed"),
        ("run-done", "completed"),
    ):
        run = await _row(manager, run_id)
        assert run.status == expected
    assert (await _row(manager, "run-stale-a")).terminal_reason == "server_restart"
    # 孤儿 run 的事件流必须有合成 done 收尾，否则历史详情页 SSE 无限重连。
    async with manager.session_factory() as session:
        orphans = (
            await session.scalars(
                select(RunEvent).where(
                    RunEvent.run_id.in_(("run-stale-a", "run-stale-b", "run-stale-c")),
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


async def test_run_graph_passes_thread_id_config(manager):
    """①恢复基建的接缝：graph 必须收到 thread_id=run_id 的 config，
    否则 checkpointer 无处落、resume 无从谈起。"""
    graph = FakeGraph(result=_completed_result())
    manager.holder["graph"] = graph
    run_id = await manager.start(USER_ID, "q")
    await _settle(manager, run_id)
    assert graph.seen_configs, "astream 未收到 kwargs"
    configs = [c.get("config") for c in graph.seen_configs]
    assert any(config["configurable"] == {"thread_id": run_id} for config in configs)
    assert all(config["callbacks"] for config in configs)


async def test_streaming_preview_routing_and_ephemerality(manager):
    """官方 flag 形态：只有 supervisor 直下（ns 深度1）的 text 进预览；
    深层嵌套（tools 路径）、writer、空文本一律静默；帧无 seq、不落库。"""
    gate = asyncio.Event()
    manager._worker._max_running = 1  # noqa: SLF001 - 订阅必须先于纯临时帧
    manager.holder["graph"] = FakeGraph(
        result=_completed_result(),
        gate=gate,
        stream_messages=[
            (("supervisor:aaa",), "先梳理缺口，"),  # ✓ 唯一放行
            (("supervisor:aaa", "tools:bbb"), "researcher串流"),  # ✗ 深度>1
            (("writer:ccc",), "writer字幕"),  # ✗ 非白名单
            (("supervisor:ddd",), ""),  # ✗ 空文本
            (
                ("supervisor:eee",),
                '[系统工具执行结果] {"active_evidence": []}',
                "tool",
            ),  # ✗ 工具回执
        ],
    )
    run_id = await manager.start(USER_ID, "q")
    _, queue = manager.fanout.subscribe(run_id)  # start 已登记任务但未开跑：必然先于 delta
    gate.set()
    await _settle(manager, run_id)

    frames = []
    while not queue.empty():
        item = queue.get_nowait()
        if item is not CLOSE_STREAM_FLAG:
            frames.append(item)
    deltas = [f for f in frames if f["event_type"] == "text_delta"]
    assert [d["payload"] for d in deltas] == [{"channel": "supervisor", "text": "先梳理缺口，"}]
    assert all("seq" not in d for d in deltas)  # ephemeral：不占 seq
    async with manager.session_factory() as session:
        persisted = (
            await session.scalars(
                select(RunEvent).where(
                    RunEvent.run_id == run_id, RunEvent.event_type == "text_delta"
                )
            )
        ).all()
    assert persisted == []
    assert (await _row(manager, run_id)).status == "completed"  # values 根命名空间取回终态


class FakeSaver:
    """triage 用的假 checkpointer：只有 alive 集合内的 thread 存在断点。"""

    def __init__(self, alive):
        self.alive = set(alive)
        self.queried = []

    async def aget_tuple(self, config):
        thread_id = config["configurable"]["thread_id"]
        self.queried.append(thread_id)
        return object() if thread_id in self.alive else None


async def test_resume_answer_is_durable_and_duplicate_submission_is_rejected(manager):
    gate = asyncio.Event()
    manager._worker._max_running = 1  # noqa: SLF001 - keep resumed work queued
    manager.holder["graph"] = FakeGraph(gate=gate, result=_completed_result())
    active = await manager.start(USER_ID, "occupy slot")
    while (await _row(manager, active)).status != "running":
        await asyncio.sleep(0.01)

    run_id = "run-awaiting-durable"
    async with manager.session_factory() as session:
        session.add(Run(id=run_id, user_id=USER_ID, query="clarify", status="awaiting_input"))
        await session.commit()
    manager._checkpointer = FakeSaver({run_id})  # noqa: SLF001

    assert await manager.resume_with_input(USER_ID, run_id, "选择第二项") == "queued"
    queued = await _row(manager, run_id)
    assert queued.resume_payload == {"answer": "选择第二项"}
    with pytest.raises(RuntimeError):
        await manager.resume_with_input(USER_ID, run_id, "重复回答")

    await manager.cancel(USER_ID, run_id)
    gate.set()
    await _settle(manager, active)


async def test_resume_triage_continues_seq_and_revives_checkpoint_run(tmp_path):
    """②全链路：分诊(判死/复活) → seq 续号 → astream(None) 续跑 → 新事件落库。"""
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path / 'resume.db'}")
    await init_db(engine)
    factory = make_session_factory(engine)
    async with factory() as session:
        session.add(User(id=USER_ID, email="r@test", password_hash="h"))
        session.add(
            Run(
                id="run-orphan",
                user_id=USER_ID,
                query="孤而可活",
                status="interrupted",
                terminal_reason="server_shutdown",
                event_seq=5,
            )
        )
        session.add(Run(id="run-dead", user_id=USER_ID, query="无据可活", status="running"))
        for i in range(1, 6):  # 上一世留下的 seq 1..5
            session.add(
                RunEvent(run_id="run-orphan", seq=i, event_type=f"engine_{i}", record={"seq": i})
            )
        await session.commit()

    holder: dict = {}
    gate = asyncio.Event()

    def graph_factory(*, settings, event_sink, http_client, checkpointer=None, tool_cache=None):
        del tool_cache
        graph = FakeGraph(
            result=_completed_result(),
            emit_events=3,
            resume_run_id="run-orphan",
            gate=gate,
        )
        graph._sink = event_sink
        holder["graph"] = graph
        return graph

    manager = RunManager(
        settings=_settings(tmp_path),
        session_factory=factory,
        config=ServiceConfig(
            database_url="unused",
            jwt_secret="s" * 40,
            service_log_dir=tmp_path,
            jsonl_events=False,
        ),
        fanout=FanoutSink(asyncio.get_running_loop()),
        http_client=SimpleNamespace(),
        graph_factory=graph_factory,
        checkpointer=FakeSaver({"run-orphan"}),
    )
    killed, resumable = await manager.reconcile_startup()
    assert killed == 1
    assert resumable == [("run-orphan", USER_ID, "孤而可活")]
    async with factory() as session:
        assert (await session.get(Run, "run-dead")).status == "failed"
        assert (await session.get(Run, "run-orphan")).status == "interrupted"  # 未判死

    assert await manager.resume_runs(resumable) == 1
    task = manager._tasks.get("run-orphan")  # noqa: SLF001
    assert task is not None
    gate.set()
    await task

    assert holder["graph"].none_inputs == 1  # 续跑用 astream(None)
    async with factory() as session:
        seqs = list(
            (
                await session.scalars(
                    select(RunEvent.seq)
                    .where(RunEvent.run_id == "run-orphan")
                    .order_by(RunEvent.seq)
                )
            ).all()
        )
        # 旧 1..5 → resuming=6 → engine×3=7..9 → run_done=10：跨世连续、无主键冲突
        assert seqs == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        resuming = await session.get(RunEvent, ("run-orphan", 6))
        assert resuming.record["payload"]["status"] == "resuming"  # 6 号帧确为续跑播报
        run = await session.get(Run, "run-orphan")
    assert run.status == "completed"
    assert run.report_markdown.startswith("# 研究报告")
    await engine.dispose()
    await asyncio.sleep(0.1)
