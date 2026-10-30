import asyncio
import logging
import re
import time
import traceback
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
import pytest_asyncio
from langgraph.types import Command
from sqlalchemy import select

from deepresearcher.config import (
    AgentConfig,
    AppConfig,
    LLMConfig,
    ObservabilityConfig,
    SearchConfig,
    Settings,
)
from deepresearcher.service.coordination.signals import PostgresSignalBus
from deepresearcher.service.events.buffer import RunEventBuffer
from deepresearcher.service.events.store import RunEventStore
from deepresearcher.service.execution.runtime import WorkerRuntime
from deepresearcher.service.preview.local import LocalPreviewBus
from deepresearcher.service.runs.admission import QuotaExceededError, RunAdmission
from deepresearcher.service.runs.manager import RunManager
from deepresearcher.service.runs.store import RunStore
from deepresearcher.service.settings import ServiceConfig
from deepresearcher.service.storage.engine import make_engine, make_session_factory
from deepresearcher.service.storage.tables import Run, RunEvent, User
from messaging_stub import InProcessDispatchBus
from pg_backend import database_url_for

pytestmark = pytest.mark.asyncio

USER_ID = 1

# 等待 run 进入终态的上限：满负载下数据库领取与建表明显变慢，2 至 3 秒不足。
_TERMINAL_WAIT_SECONDS = 20.0


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
        early_values=None,
        resume_run_id="run-resume",
        interrupt_payload=None,
    ):
        self.result = result or {}
        self.error = error
        self.gate = gate
        self.emit_events = emit_events
        self.resume_run_id = resume_run_id
        self.stream_messages = list(stream_messages)  # [(namespace_tuple, text)]
        # 模拟 clarify 超步后携带 run_headline 的根 values(在结果之前发出)。
        self.early_values = dict(early_values or {})
        self.ainvoke_inputs: list[dict] = []
        self.seen_configs: list[dict] = []
        self.none_inputs = 0
        self.interrupt_payload = interrupt_payload

    async def _run(self, input):  # noqa: A002 - 与 LangGraph 契约同名
        # resume 有两种形态：astream(None)（系统重启续跑）与 astream(Command(resume=…))
        # （澄清回答续跑）；两者都不是 dict，都不能按输入字典取 run_id。
        if not isinstance(input, dict) or not input:
            run_id = self.resume_run_id
            self.none_inputs += 1
        else:
            run_id = input["run_id"]
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
        """复刻官方形态:subgraphs=True 时 yield (namespace, mode, chunk)。

        stream_messages 条目 = (namespace, text) 或 (namespace, text, msg_type)。
        langchain 1.x 真实流式片段的 .type 是 "AIMessageChunk",若假图默认成
        "ai" 会让执行器的放行判断与生产脱钩,流式问题测不出来。
        "tool" 模拟 messages 模式里混入的工具回执。
        """
        self.seen_configs.append(dict(kwargs))  # 记录 config（thread_id 注入断言用）
        if self.early_values:
            yield (), "values", dict(self.early_values)
        for entry in self.stream_messages:
            namespace, text = entry[0], entry[1]
            msg_type = entry[2] if len(entry) > 2 else "AIMessageChunk"
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


class ServiceHarness:
    """在集成风格测试中显式暴露控制层与执行层。"""

    def __init__(
        self,
        *,
        controller,
        execution,
        config,
        engine,
        session_factory,
        buffer,
        preview,
        holder,
        signal_bus,
        bus,
    ):
        self.controller = controller
        self.execution = execution
        self.config = config
        self.engine = engine
        self.session_factory = session_factory
        self.buffer = buffer
        self.preview = preview
        self.holder = holder
        self.signal_bus = signal_bus
        self.bus = bus

    async def start(self, user_id, query):
        return await self.controller.start(user_id, query)

    async def cancel(self, user_id, run_id):
        return await self.controller.cancel(user_id, run_id)

    async def resume_with_input(self, user_id, run_id, answer):
        return await self.controller.resume_with_input(user_id, run_id, answer)

    async def shutdown(self):
        await self.execution.shutdown()
        await self.signal_bus.close()


@pytest_asyncio.fixture
async def manager(tmp_path, request):
    # 文件库而非 :memory: + StaticPool：StaticPool 全共享一条连接，后台任务与
    # cancel() 的并发 session 会互相干扰，一方回滚会 terminate 另一方在用的连接。
    # 设置 SERVICE_TEST_DATABASE_URL 时这一层改用真实 PG。
    engine = make_engine(database_url_for(tmp_path))
    session_factory = make_session_factory(engine)
    async with session_factory() as session:
        from deepresearcher.service.storage.tables import User

        session.add(User(id=USER_ID, email="u@test", password_hash="h"))
        await session.commit()
    preview = LocalPreviewBus()
    # 槽位是唯一的执行容量闸门：用例要单槽时按参数收紧，而不是事后改消费者字段。
    max_running = getattr(request, "param", 3)
    config = ServiceConfig(
        database_url="unused",
        jwt_secret="s" * 40,
        service_log_dir=tmp_path,
        jsonl_events=False,
        max_concurrent_runs_per_user=2,
        worker_max_running_runs=max_running,
    )
    holder: dict = {}

    def graph_factory(
        *,
        settings,
        event_sink,
        http_client,
        checkpointer=None,
        material_store=None,
        provider_health=None,
    ):
        del material_store
        holder["sink"] = event_sink
        holder["checkpointer"] = checkpointer
        graph = holder.get("graph") or FakeGraph()
        graph._sink = event_sink
        return graph

    signal_bus = PostgresSignalBus()
    event_store = RunEventStore(session_factory)
    buffer = RunEventBuffer(store=event_store, signal_bus=signal_bus)
    bus = InProcessDispatchBus()
    execution = WorkerRuntime(
        settings=_settings(tmp_path),
        session_factory=session_factory,
        config=config,
        buffer=buffer,
        http_client=SimpleNamespace(),
        publisher=bus,
        receiver=bus,
        graph_factory=graph_factory,
        ephemeral_bus=preview,
    )
    controller = RunManager(
        session_factory=session_factory,
        config=config,
        buffer=buffer,
        publisher=bus,
        signal_bus=signal_bus,
    )
    signal_bus.subscribe("run_cancel_requested", execution.handle_cancel_notification)
    # 消息驱动：必须先让拉取循环跑起来，受理后的分派消息才会被取走执行。
    await execution.start()
    manager = ServiceHarness(
        controller=controller,
        execution=execution,
        config=config,
        engine=engine,
        session_factory=session_factory,
        buffer=buffer,
        preview=preview,
        holder=holder,
        signal_bus=signal_bus,
        bus=bus,
    )
    yield manager
    # 先结束所有后台任务（它们的 finally 还要写库），再拆引擎，避免
    # “closed database / no such table” 竞态。
    await manager.shutdown()
    await engine.dispose()


async def _settle(manager, run_id):
    async def _await_task():
        # 消息驱动下任务不是立刻存在的：受理后要等拉取循环取走消息、acquire 成功。
        while run_id not in manager.execution.tasks:
            await asyncio.sleep(0.01)
        return manager.execution.tasks[run_id]

    try:
        task = await asyncio.wait_for(_await_task(), timeout=5)
    except TimeoutError:
        return
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


async def _event_types(manager, run_id):
    """该 run 已写入的事件类型，按 seq 排序。"""
    async with manager.session_factory() as session:
        return list(
            (
                await session.scalars(
                    select(RunEvent.event_type)
                    .where(RunEvent.run_id == run_id)
                    .order_by(RunEvent.seq)
                )
            ).all()
        )


async def _status_frames(manager, run_id):
    """该 run 已落库的 run_status 帧里的状态序列。"""
    async with manager.session_factory() as session:
        records = (
            await session.scalars(
                select(RunEvent.record)
                .where(RunEvent.run_id == run_id, RunEvent.event_type == "run_status")
                .order_by(RunEvent.seq)
            )
        ).all()
    return [record["payload"]["status"] for record in records]


async def _done_frames(manager, run_id):
    async with manager.session_factory() as session:
        return (
            await session.scalars(
                select(RunEvent).where(RunEvent.run_id == run_id, RunEvent.event_type == "run_done")
            )
        ).all()


async def test_manager_delegates_execution_without_building_graph_itself(manager):
    captured = {}
    original_execute = manager.execution.executor.execute

    async def execute(work, **kwargs):
        captured.update(work=work, kwargs=kwargs)
        await original_execute(work, **kwargs)

    manager.execution.executor.execute = execute
    run_id = await manager.start(USER_ID, "  委托执行  ")
    await _settle(manager, run_id)

    assert captured["work"].run_id == run_id
    assert captured["work"].user_id == USER_ID
    assert captured["work"].query == "委托执行"
    assert captured["work"].claimed is True
    assert captured["kwargs"]["resume_input"] is None


async def test_running_status_is_announced_only_for_user_visible_claim(manager):
    """人工回答恢复要播 running 帧;重启恢复的续跑不播。"""
    manager.holder["graph"] = FakeGraph(result=_completed_result())
    async with manager.session_factory() as session:
        session.add_all(
            [
                Run(id="run-system-resume", user_id=USER_ID, query="system", status="interrupted"),
                Run(
                    id="run-human-resume",
                    user_id=USER_ID,
                    query="human",
                    status="queued",
                    resume_payload={"answer": "成本"},
                ),
            ]
        )
        await session.commit()

    store = manager.execution.store
    executor = manager.execution.executor

    system_work = await store.acquire("run-system-resume", worker_id="w-test")
    assert system_work is not None and system_work.resume is True
    await executor.execute(system_work)
    assert await _status_frames(manager, "run-system-resume") == []

    # 人工回答恢复：answer_resume 已把行退回 queued 并带上载荷，领取后必须补一帧
    # running，否则前端停在 queued 直到第一个阶段事件。
    human_work = await store.acquire("run-human-resume", worker_id="w-test")
    assert human_work is not None and human_work.resume is True
    await executor.execute(human_work, resume_input=Command(resume={"answer": "成本"}))
    assert await _status_frames(manager, "run-human-resume") == ["running"]


async def test_success_persists_terminal_and_injects_run_id(manager):
    graph = FakeGraph(result=_completed_result())
    manager.holder["graph"] = graph
    run_id = await manager.start(USER_ID, "  测试问题  ")
    await _settle(manager, run_id)

    # run_id/session_id 必须显式进入引擎输入
    assert graph.ainvoke_inputs[0] == {"query": "测试问题", "run_id": run_id, "session_id": run_id}
    run = await _row(manager, run_id)
    assert run.status == "completed"
    assert run.terminal_reason == "report_rendered"
    assert run.report_markdown.startswith("# 研究报告")
    assert run.citations_json == graph.result["citations"]
    assert (run.evidence_count, run.source_count) == (41, 12)
    assert run.query == "测试问题"
    async with manager.session_factory() as session:
        trace_events = list(
            (
                await session.scalars(
                    select(RunEvent)
                    .where(
                        RunEvent.run_id == run_id,
                        RunEvent.event_type.in_(("trace_started", "trace_completed")),
                    )
                    .order_by(RunEvent.seq)
                )
            ).all()
        )
    assert [event.event_type for event in trace_events] == ["trace_started", "trace_completed"]
    assert trace_events[0].record["trace_id"] == trace_events[1].record["trace_id"]
    assert re.match(r"^[0-9a-f]{32}$", trace_events[0].record["trace_id"])
    assert trace_events[0].record["metadata"]["resume"] is False


async def test_failure_persists_failed_with_truncated_message(manager):
    manager.holder["graph"] = FakeGraph(error=RuntimeError("x" * 900))
    run_id = await manager.start(USER_ID, "q")
    await _settle(manager, run_id)
    run = await _row(manager, run_id)
    assert run.status == "failed"
    assert run.terminal_reason == "run_exception"
    assert len(run.error_message) <= 500


async def test_llm_auth_error_fails_fast_with_typed_reason(manager):
    class APIError(Exception):  # 模仿 openai.APIStatusError：带 status_code
        status_code = 401

    manager.holder["graph"] = FakeGraph(error=APIError("Invalid API key"))
    run_id = await manager.start(USER_ID, "q")
    await _settle(manager, run_id)
    run = await _row(manager, run_id)
    assert run.status == "failed"
    assert run.terminal_reason == "llm_unavailable:invalid_key"
    # 面向用户的安全文案，不含原始内部错误串
    assert "密钥" in run.error_message and "Invalid API key" not in run.error_message


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
    assert len(dones) == 1  # done 恰一次，且必须落 DB 以便重连回放


async def test_remote_manager_cancel_uses_notification_without_waiting_for_lease_check(manager):
    gate = asyncio.Event()
    manager.execution.consumer._lease_check_seconds = 20  # noqa: SLF001 - 验证快路径不依赖租约确认
    manager.holder["graph"] = FakeGraph(gate=gate, result=_completed_result())
    run_id = await manager.start(USER_ID, "remote cancel")
    while (await _row(manager, run_id)).status != "running":
        await asyncio.sleep(0.01)

    signal_bus = PostgresSignalBus()
    subscription = signal_bus.subscribe(
        "run_cancel_requested", manager.execution.handle_cancel_notification
    )
    try:
        controller = RunManager(
            session_factory=manager.session_factory,
            config=manager.config,
            buffer=RunEventBuffer(
                store=RunEventStore(manager.session_factory),
                signal_bus=signal_bus,
            ),
            publisher=manager.bus,
            signal_bus=signal_bus,
        )
        started = time.monotonic()
        requested = await controller.cancel(USER_ID, run_id)
        assert requested.status == "running"
        assert requested.cancellation_requested_at is not None
        await _settle(manager, run_id)
        assert time.monotonic() - started < 1.0
        assert (await _row(manager, run_id)).status == "cancelled"
    finally:
        signal_bus.unsubscribe("run_cancel_requested", subscription)
        await signal_bus.close()


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
    while run_id not in manager.execution.tasks or manager.execution.tasks[run_id].done():
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


async def test_real_suspend_then_resume_matches_is_null_cas(manager):
    """回归守卫:真实挂起时 worker claim 分支写 resume_payload=None,必须让
    resume 的 `is_(None)` CAS 命中。

    JSON 列默认把 None 存成 JSON 字面量 null,Python 读回同样是 None,代码里
    看不出差别,但 SQL 的 IS NULL 恒为 false,澄清回答会一直 409 无法恢复。
    因此断言必须在 SQL 层用 is_(None),不能比对 Python None。
    """
    manager.holder["graph"] = FakeGraph(
        result=_completed_result(),
        interrupt_payload={"kind": "clarification", "question": "哪方面？", "options": ["成本"]},
    )
    run_id = await manager.start(USER_ID, "房价值得投资吗")
    await _settle(manager, run_id)
    assert (await _row(manager, run_id)).status == "awaiting_input"

    # SQL 层不变式：挂起后载荷必须是真正的 SQL NULL，而非 JSON null;
    # 用 Python 侧比对看不出这个差别。
    async with manager.session_factory() as session:
        is_sql_null = await session.scalar(
            select(Run.resume_payload.is_(None)).where(Run.id == run_id)
        )
    assert is_sql_null is True, "挂起写入把 None 存成了 JSON null，resume 的 CAS 将永远落空"

    # 换一张能正常结束的图，让 resume→claim 后本次执行跑到 completed，避免挂起态残留。
    manager.holder["graph"] = FakeGraph(result=_completed_result(), resume_run_id=run_id)
    manager.controller._checkpointer = FakeSaver({run_id})  # noqa: SLF001 - resume 需要断点在
    advanced = await manager.resume_with_input(USER_ID, run_id, "成本")
    assert (
        advanced != "awaiting_input"
    )  # CAS 落空时此处抛 RuntimeError already_resumed、回答永远 409
    await _settle(manager, run_id)
    assert (await _row(manager, run_id)).status == "completed"


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
    service = RunAdmission(
        session_factory=manager.session_factory,
        config=ServiceConfig(
            database_url="unused",
            jwt_secret="s" * 40,
            max_concurrent_runs_per_user=10,
            max_global_queued_runs=1,
        ),
        publisher=manager.bus,
    )
    await service.create(USER_ID, "q1")
    with pytest.raises(QuotaExceededError, match="等待队列已满"):
        await service.create(USER_ID, "q2")


async def test_two_store_instances_serialize_claims_on_the_same_row(manager):
    # 直接插行而不走受理：受理会顺带发一条分派消息，后台消费者会先一步领走，
    # 这条用例要验的是两个领取者对同一行的条件 UPDATE，不该被分派扰乱。
    run_id = "run-atomic-claim"
    async with manager.session_factory() as session:
        session.add(Run(id=run_id, user_id=USER_ID, query="atomic claim", status="queued"))
        await session.commit()
    first_store = RunStore(manager.session_factory)
    second_store = RunStore(manager.session_factory)

    claims = await asyncio.gather(
        first_store.acquire(run_id, worker_id="worker-a"),
        second_store.acquire(run_id, worker_id="worker-b"),
    )

    # 两次领取都被行锁串行化：世代严格递增，只有一个能写。
    assert all(claim is not None for claim in claims)
    assert sorted(claim.fence_token for claim in claims) == [1, 2]
    owners = [claim for claim in claims if await first_store.verify_lease(claim) == "held"]
    assert len(owners) == 1
    assert owners[0].fence_token == 2


async def test_worker_runs_admission_from_another_process(manager, caplog):
    """独立 Worker 不依赖调用方唤醒，受理落库后的消息由它自己拉取。"""
    caplog.set_level(logging.WARNING)
    graph = FakeGraph(result=_completed_result(), emit_events=1)
    manager.holder["graph"] = graph
    service = RunAdmission(
        session_factory=manager.session_factory, config=manager.config, publisher=manager.bus
    )
    run_id = await service.create(USER_ID, "external admission")

    await manager.execution.start()
    async with asyncio.timeout(_TERMINAL_WAIT_SECONDS):
        while (await _row(manager, run_id)).status != "completed":
            await asyncio.sleep(0.01)
    await _settle(manager, run_id)

    assert len(graph.ainvoke_inputs) == 1
    # 终态写入早于事件 flush（flush 在 task 收尾）；跨进程受理时 task 可能已从 tasks
    # 表移除，因此以事件出现为判据，而不是假定它已写入。
    try:
        async with asyncio.timeout(_TERMINAL_WAIT_SECONDS):
            while "engine_0" not in await _event_types(manager, run_id):
                await asyncio.sleep(0.02)
    except TimeoutError:
        row = await _row(manager, run_id)
        pytest.fail(
            f"engine 事件未写入：status={row.status} event_seq={row.event_seq} "
            f"open={manager.buffer.is_open(run_id)} "
            f"pending={manager.buffer._pending.get(run_id, [])!r} "  # noqa: SLF001
            f"tasks={sorted(manager.execution.tasks)}\n{caplog.text}"
        )


async def test_takeover_retires_the_previous_fence(manager):
    """接管之后旧世代只剩读权：它的写库一律打空，行状态由新持有者决定。"""
    run_id = "run-owner-cas"
    async with manager.session_factory() as session:
        session.add(Run(id=run_id, user_id=USER_ID, query="owner CAS", status="queued"))
        await session.commit()
    store = RunStore(manager.session_factory)
    stale = await store.acquire(run_id, worker_id="worker-a")
    assert stale is not None

    fresh = await store.acquire(run_id, worker_id="worker-b")
    assert fresh is not None
    assert await store.verify_lease(stale) == "lost"

    assert await manager.execution.executor.persist_status(stale, status="completed") is None
    assert (await _row(manager, run_id)).status == "running"
    assert await manager.execution.executor.persist_status(fresh, status="completed") is not None


async def test_two_event_stores_allocate_non_overlapping_sequences(manager):
    # 同上：直接插行保证事件表是空的，序号分配才从 1 开始可断言。
    run_id = "run-event-sequence"
    async with manager.session_factory() as session:
        session.add(Run(id=run_id, user_id=USER_ID, query="event sequence", status="queued"))
        await session.commit()
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


@pytest.mark.parametrize("manager", [1], indirect=True)
async def test_global_capacity_keeps_excess_run_queued_then_dispatches(manager):
    gate = asyncio.Event()
    manager.holder["graph"] = FakeGraph(gate=gate, result=_completed_result())

    first = await manager.start(USER_ID, "q1")
    second = await manager.start(USER_ID, "q2")

    while (await _row(manager, first)).status != "running":
        await asyncio.sleep(0.01)
    assert (await _row(manager, second)).status == "queued"
    assert second not in manager.execution.tasks

    gate.set()
    await _settle(manager, first)
    for _ in range(100):
        if (await _row(manager, second)).status != "queued":
            break
        await asyncio.sleep(0.01)
    await _settle(manager, second)
    assert (await _row(manager, second)).status == "completed"


@pytest.mark.parametrize("manager", [1], indirect=True)
async def test_cancel_queued_run_never_executes_graph(manager):
    gate = asyncio.Event()
    graph = FakeGraph(gate=gate, result=_completed_result())
    manager.holder["graph"] = graph

    first = await manager.start(USER_ID, "q1")
    queued = await manager.start(USER_ID, "q2")
    assert (await _row(manager, queued)).status == "queued"

    cancelled = await manager.cancel(USER_ID, queued)
    assert cancelled.status == "cancelled"
    assert queued not in manager.execution.tasks

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
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    assert types[0] == "run_status"  # queued 最早
    assert types[-1] == "run_done"  # done 严格最后（重连回放以此收尾）
    assert "engine_0" in types and "engine_2" in types
    assert types.count("run_done") == 1


async def test_headline_persists_at_clarify_exit_not_at_end(manager):
    """headline 落库时机:携带 run_headline 的根 values 在 clarify 超步出口一到
    就落库推帧,不等 run 终态;终值再含 headline 时由 CAS 挡住,只落一次。"""
    gate = asyncio.Event()
    manager.holder["graph"] = FakeGraph(
        result=_completed_result(),
        gate=gate,
        early_values={"run_headline": "澄清出口即写"},
    )
    run_id = await manager.start(USER_ID, "q")
    async with asyncio.timeout(5):
        while (await _row(manager, run_id)).headline is None:
            await asyncio.sleep(0.02)
    row = await _row(manager, run_id)
    assert row.headline == "澄清出口即写"
    assert row.status == "running"  # 还没到终态，标题已在

    # 帧必须在 gate 打开前落库——落库→flush→NOTIFY 一次走完，不等周期。
    # 写行与写帧是两笔事务（前者是 CAS 成功的前置条件），因此这里等到帧出现为止，
    # 而不是刚看到行就断言：真正要证的是「澄清出口就发帧」，不是「同一瞬间已落库」。
    async def _headline_frames() -> list[RunEvent]:
        async with manager.session_factory() as session:
            return list(
                (
                    await session.scalars(
                        select(RunEvent).where(
                            RunEvent.run_id == run_id,
                            RunEvent.event_type == "run_headline_updated",
                        )
                    )
                ).all()
            )

    async with asyncio.timeout(5):
        frames = await _headline_frames()
        while not frames:
            await asyncio.sleep(0.02)
            frames = await _headline_frames()
    assert not gate.is_set()  # 帧到得比终态早：图还卡在 gate 上
    assert [f.record["payload"]["headline"] for f in frames] == ["澄清出口即写"]
    gate.set()
    await _settle(manager, run_id)
    # 终值再含 headline 也只此一次：CAS 与 headline_persisted 双重保证不重复推帧。
    async with manager.session_factory() as session:
        frames_after = (
            await session.scalars(
                select(RunEvent).where(
                    RunEvent.run_id == run_id,
                    RunEvent.event_type == "run_headline_updated",
                )
            )
        ).all()
    assert len(frames_after) == 1


async def test_run_graph_passes_thread_id_config(manager):
    """恢复的接缝:graph 必须收到 thread_id=run_id 的 config,
    否则 checkpointer 无处落、resume 无法续跑。"""
    graph = FakeGraph(result=_completed_result())
    manager.holder["graph"] = graph
    run_id = await manager.start(USER_ID, "q")
    await _settle(manager, run_id)
    assert graph.seen_configs, "astream 未收到 kwargs"
    configs = [c.get("config") for c in graph.seen_configs]
    assert any(config["configurable"] == {"thread_id": run_id} for config in configs)
    assert all(config["callbacks"] for config in configs)


async def test_streaming_preview_routing_and_ephemerality(manager):
    """只有 supervisor 直下、命名空间深度 1 的 text 进预览;tools 深层嵌套、
    writer、空文本都不出口;帧不带 seq、不落库。

    预览只有一条路径:执行器直发注入的总线。测试里 manager 与 executor
    共享同一个 LocalPreviewBus 实例。"""
    gate = asyncio.Event()
    manager.holder["graph"] = FakeGraph(
        result=_completed_result(),
        gate=gate,
        stream_messages=[
            (("supervisor:aaa",), "先梳理缺口，"),  # ✓ 唯一放行(真实 AIMessageChunk 形态)
            (
                ("supervisor:fff",),
                "旧版聚合消息也放行",
                "ai",
            ),  # ✓ 兼容非流式回退的 AIMessage.type
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
    # start 已登记任务但未开跑：订阅必然先于 delta
    subscription = await manager.preview.subscribe(run_id)
    gate.set()
    await _settle(manager, run_id)

    frames = []
    while not subscription.queue.empty():
        frames.append(subscription.queue.get_nowait())
    deltas = [f for f in frames if f["event_type"] == "text_delta"]
    assert [d["payload"] for d in deltas] == [
        {"channel": "supervisor", "text": "先梳理缺口，"},
        {"channel": "supervisor", "text": "旧版聚合消息也放行"},
    ]
    assert all("seq" not in d for d in deltas)  # 临时帧，不占 seq
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
    manager.holder["graph"] = FakeGraph(gate=gate, result=_completed_result())
    active = await manager.start(USER_ID, "occupy slot")
    while (await _row(manager, active)).status != "running":
        await asyncio.sleep(0.01)

    run_id = "run-awaiting-durable"
    async with manager.session_factory() as session:
        session.add(Run(id=run_id, user_id=USER_ID, query="clarify", status="awaiting_input"))
        await session.commit()
    manager.controller._checkpointer = FakeSaver({run_id})  # noqa: SLF001 - 注入假断点存储

    assert await manager.resume_with_input(USER_ID, run_id, "选择第二项") == "queued"
    queued = await _row(manager, run_id)
    assert queued.resume_payload == {"answer": "选择第二项"}
    with pytest.raises(RuntimeError):
        await manager.resume_with_input(USER_ID, run_id, "重复回答")

    await manager.cancel(USER_ID, run_id)
    gate.set()
    await _settle(manager, active)


async def test_crash_takeover_resumes_from_checkpoint(tmp_path):
    """崩溃接管全链路：重投到达 → fence 前进 → astream(None) 续跑 → seq 续号。"""
    engine = make_engine(database_url_for(tmp_path))
    factory = make_session_factory(engine)
    async with factory() as session:
        session.add(User(id=USER_ID, email="r@test", password_hash="h"))
        session.add(
            Run(
                id="run-orphan",
                user_id=USER_ID,
                query="孤而可活",
                status="running",
                lease_owner="dead-worker",
                fence_token=1,
                event_seq=5,
            )
        )
        for i in range(1, 6):  # 上一个进程已写入的 seq 1..5
            session.add(
                RunEvent(run_id="run-orphan", seq=i, event_type=f"engine_{i}", record={"seq": i})
            )
        await session.commit()

    holder: dict = {}
    gate = asyncio.Event()

    def graph_factory(
        *,
        settings,
        event_sink,
        http_client,
        checkpointer=None,
        material_store=None,
        provider_health=None,
    ):
        del material_store
        graph = FakeGraph(
            result=_completed_result(),
            emit_events=3,
            resume_run_id="run-orphan",
            gate=gate,
        )
        graph._sink = event_sink
        holder["graph"] = graph
        return graph

    config = ServiceConfig(
        database_url="unused",
        jwt_secret="s" * 40,
        service_log_dir=tmp_path,
        jsonl_events=False,
    )
    event_store = RunEventStore(factory)
    buffer = RunEventBuffer(store=event_store, signal_bus=PostgresSignalBus())
    bus = InProcessDispatchBus()
    execution = WorkerRuntime(
        settings=_settings(tmp_path),
        session_factory=factory,
        config=config,
        buffer=buffer,
        http_client=SimpleNamespace(),
        publisher=bus,
        receiver=bus,
        graph_factory=graph_factory,
        checkpointer=FakeSaver({"run-orphan"}),
    )
    await execution.start()
    # 崩溃的进程不会再确认那条消息：可见性到期后 Broker 重投，这里就是那次重投。
    await bus.redispatch("run-orphan")
    async with asyncio.timeout(5):
        while "run-orphan" not in execution.tasks:
            await asyncio.sleep(0.01)
    task = execution.tasks["run-orphan"]
    gate.set()
    await task

    assert holder["graph"].none_inputs == 1  # 续跑用 astream(None)
    async with factory() as session:
        persisted = list(
            (
                await session.scalars(
                    select(RunEvent).where(RunEvent.run_id == "run-orphan").order_by(RunEvent.seq)
                )
            ).all()
        )
        # 旧 1..5 → resuming=6 → trace + engine + done：跨进程连续，无主键冲突。
        assert [event.seq for event in persisted] == list(range(1, 13))
        assert [event.event_type for event in persisted[6:]] == [
            "trace_started",
            "engine_0",
            "engine_1",
            "engine_2",
            "trace_completed",
            "run_done",
        ]
        resuming = await session.get(RunEvent, ("run-orphan", 6))
        assert resuming.record["payload"]["status"] == "resuming"  # 6 号帧确为续跑播报
        run = await session.get(Run, "run-orphan")
    assert run.status == "completed"
    assert run.report_markdown.startswith("# 研究报告")
    await execution.shutdown()
    await engine.dispose()
    await asyncio.sleep(0.1)


async def test_takeover_without_checkpoint_restarts_from_scratch(manager):
    """接管一个没有断点的行：降级为从头跑。

    传 None 给 astream 会抛 EmptyInputError，把一个重跑成本≈0 的任务
    变成必然 failed——那种结局比从头跑差得多。
    """
    run_id = "run-no-checkpoint"
    async with manager.session_factory() as session:
        session.add(
            Run(
                id=run_id,
                user_id=USER_ID,
                query="无据可活",
                status="running",
                lease_owner="dead-worker",
                fence_token=1,
            )
        )
        await session.commit()
    manager.holder["graph"] = FakeGraph(result=_completed_result(), emit_events=1)

    await manager.bus.redispatch(run_id)
    async with asyncio.timeout(_TERMINAL_WAIT_SECONDS):
        while (await _row(manager, run_id)).status != "completed":
            await asyncio.sleep(0.01)
    await _settle(manager, run_id)

    assert manager.holder["graph"].ainvoke_inputs  # 走的是 {"query": ...}，不是 None
    assert manager.holder["graph"].none_inputs == 0


async def test_cancel_intent_on_running_row_settles_on_redelivery(manager):
    """运行中点取消、持有者又已死：重投到达时就地结算，并补 done 帧。

    这条路径不能等执行器自己收尾（它已经不在了），也不能等扫描器
    （单租约下没有扫描器）。
    """
    run_id = "run-cancel-sunk"
    async with manager.session_factory() as session:
        session.add(
            Run(
                id=run_id,
                user_id=USER_ID,
                query="q",
                status="running",
                lease_owner="dead-worker",
                fence_token=1,
                cancellation_requested_at=datetime.now(timezone.utc),
            )
        )
        await session.commit()

    await manager.bus.redispatch(run_id)
    async with asyncio.timeout(_TERMINAL_WAIT_SECONDS):
        # 终态先落库、done 帧随后补发，因此等到 done 出现才算这条路径走完。
        while not await _done_frames(manager, run_id):
            await asyncio.sleep(0.01)

    run = await _row(manager, run_id)
    assert run.terminal_reason == "user_cancelled"
    assert run.finished_at is not None
    assert run.lease_owner is None
    assert len(await _done_frames(manager, run_id)) == 1


class _FlakyOwnershipStore:
    """前 fail_rounds 拍探针抛错，之后一直答 ok。"""

    def __init__(self, *, fail_rounds: int) -> None:
        self._fail_rounds = fail_rounds
        self.poll_calls = 0

    async def verify_lease(self, _work):
        self.poll_calls += 1
        if self.poll_calls <= self._fail_rounds:
            raise ConnectionError("connection pool hiccup")
        return "held"


class _RecordingLeaseExecutor:
    def __init__(self) -> None:
        self.lease_lost: list[str] = []

    def mark_lease_lost(self, run_id):
        self.lease_lost.append(run_id)


class _NullReceiver:
    """不投递任何消息的入站桩：这些用例只关心拉取循环与租约确认。"""

    async def receive(self):
        return None

    async def ack(self, _dispatch):
        return None

    async def extend_visibility(self, _dispatch, _seconds):
        return None

    async def close(self):
        return None


def _inbound(run_id: str):
    from deepresearcher.service.messaging.dispatch import new_dispatch
    from deepresearcher.service.messaging.receiver import DispatchDelivery

    return DispatchDelivery(message=new_dispatch(run_id), receipt=object(), delivery_attempt=0)


def _watch_worker(store, executor):
    from deepresearcher.service.execution.consumer import RunConsumer

    worker = RunConsumer(
        store=store,
        executor=executor,
        buffer=object(),  # type: ignore[arg-type] - 租约确认与拉取路径不写事件
        receiver=_NullReceiver(),
        max_running=1,
        lease_check_seconds=600,
        invisible_seconds=600,
        visibility_renew_seconds=600,
    )
    worker._lease_check_seconds = 0.01  # noqa: SLF001 - 测试提速
    return worker


async def test_lease_hold_survives_transient_db_errors():
    from deepresearcher.service.runs.store import RunWork

    work = RunWork(run_id="r1", user_id=1, query="q", lease_owner="w", fence_token=1)
    store = _FlakyOwnershipStore(fail_rounds=2)
    executor = _RecordingLeaseExecutor()
    worker = _watch_worker(store, executor)

    task = asyncio.create_task(  # noqa: SLF001 - 直接驱动租约确认循环
        worker._hold_lease(work, None, _inbound("r1"))
    )
    async with asyncio.timeout(_TERMINAL_WAIT_SECONDS):
        while store.poll_calls < 3:
            await asyncio.sleep(0.01)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert executor.lease_lost == []  # 瞬断不误杀健康执行


async def test_lease_hold_escalates_persistent_db_errors_to_lease_lost():
    from deepresearcher.service.runs.store import RunWork

    work = RunWork(run_id="r1", user_id=1, query="q", lease_owner="w", fence_token=1)
    store = _FlakyOwnershipStore(fail_rounds=99)
    executor = _RecordingLeaseExecutor()
    worker = _watch_worker(store, executor)
    worker._lease_check_failure_budget = 2  # noqa: SLF001

    async with asyncio.timeout(_TERMINAL_WAIT_SECONDS):
        await worker._hold_lease(work, None, _inbound("r1"))  # noqa: SLF001

    assert executor.lease_lost == ["r1"]  # 预算耗尽必须降级，不许静默退场


class _ScriptedReceiver:
    """receive 按剧本逐次抛错/返回;剧本耗尽后恒返回 None（长轮询空转）。"""

    def __init__(self, outcomes: list[object]) -> None:
        self._outcomes = list(outcomes)
        self.calls = 0

    async def receive(self):
        self.calls += 1
        if not self._outcomes:
            return None
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def ack(self, _dispatch):
        return None

    async def extend_visibility(self, _dispatch, _seconds):
        return None

    async def close(self):
        return None


def _scripted_worker(receiver, *, lease_check_seconds: float = 600):
    from deepresearcher.service.execution.consumer import RunConsumer

    class _NoDispatchExecutor:
        """桩路径不派发任何 run;shutdown 只用到 mark_shutdown 空集合上报。"""

        def mark_shutdown(self, _run_ids):
            return None

    return RunConsumer(
        store=object(),
        executor=_NoDispatchExecutor(),
        buffer=object(),  # type: ignore[arg-type] - 拉取循环不写事件
        receiver=receiver,
        max_running=1,
        lease_check_seconds=lease_check_seconds,
        invisible_seconds=600,
        visibility_renew_seconds=600,
    )


async def test_receive_loop_survives_transient_pull_error(monkeypatch):
    """消息队列瞬断后拉取循环继续下一跳。

    这条循环是唯一的执行入口，退出等于整个 Worker 停止消费，瞬断只退化成重试。
    """
    from deepresearcher.service.execution import consumer as consumer_module

    monkeypatch.setattr(consumer_module, "_RECEIVE_RETRY_BACKOFF_SECONDS", 0.01)
    receiver = _ScriptedReceiver([ConnectionError("messaging hiccup")])
    worker = _scripted_worker(receiver)
    receive = asyncio.create_task(worker._receive_loop())  # noqa: SLF001
    try:
        async with asyncio.timeout(_TERMINAL_WAIT_SECONDS):
            while receiver.calls < 2:
                await asyncio.sleep(0.01)
        assert not receive.done()  # 瞬断没有杀死循环，第二跳已发生
    finally:
        receive.cancel()
        await asyncio.gather(receive, return_exceptions=True)
