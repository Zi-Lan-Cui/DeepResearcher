"""派发链路：任务编号分配、在途登记、回执时序与同轮并发。"""

import asyncio
from types import SimpleNamespace

from langchain_core.messages import AIMessage

from deepresearcher.agents.supervisor import Supervisor, services
from deepresearcher.agents.supervisor.state import (
    SupervisorRoundState,
)
from deepresearcher.agents.supervisor.tools import build_supervisor_tools
from deepresearcher.config import AgentConfig
from deepresearcher.schemas import (
    PendingTask,
    TaskResult,
    TaskStopReason,
)
from supervisor_kit import (
    COMPLETE_ARGS,
    FixedResearcher,
    GatedResearcher,
    SupervisorLLM,
    delegate_context_for,
)


def test_supervisor_dispatches_independent_research_agents_concurrently():
    config = AgentConfig(
        max_research_rounds=2,
        supervisor_max_concurrent_tasks=2,
        research_agent_max_evidences_per_task=1,
    )
    supervisor = Supervisor(
        SupervisorLLM(
            delegate_topics=["方向一的事实", "方向二的事实"],
            complete_args=COMPLETE_ARGS,
        ),
        config,
        research_agent=FixedResearcher(),
    )
    result = asyncio.run(
        supervisor.run({"query": "研究问题", "clarified_query": "研究问题", "evidences": []})
    )

    assert len(result["task_results"]) == 2
    assert len(result["evidences"]) == 2
    assert result["supervisor"].status == "completed"
    assert result["supervisor_next"] == "writer"
    context = [str(message.content) for message in result["supervisor_messages"]]
    assert any("研究委托" in message for message in context)
    # 派发批回执只表受理；方向报告经 WaitTask 收据送达，含结论与该方向带回的 Evidence claim。
    assert len([message for message in context if '"status": "dispatched"' in message]) == 2
    arrivals = [
        message
        for message in context
        if '"status": "completed"' in message and '"tasks":' in message
    ]
    assert len(arrivals) == 1
    assert "task-0001" in arrivals[0] and "task-0002" in arrivals[0]
    assert all("直接事实" in message for message in arrivals)


def test_concurrent_delegates_allocate_unique_task_ids_before_absorb():
    """任务序号在分配时消耗，不从已完成结果反推 max+1。"""

    class InterleavedResearcher(FixedResearcher):
        async def run(self, task, **kwargs):
            await asyncio.sleep(0)  # 放行另一路 delegate 走到序号分配
            return await super().run(task, **kwargs)

    supervisor = Supervisor(
        SupervisorLLM(
            delegate_topics=["方向A的事实", "方向B的事实"],
            complete_args=COMPLETE_ARGS,
        ),
        AgentConfig(max_research_rounds=2, supervisor_max_concurrent_tasks=2),
        research_agent=InterleavedResearcher(),
    )
    result = asyncio.run(
        supervisor.run({"query": "研究问题", "clarified_query": "研究问题", "evidences": []})
    )

    task_ids = [
        item.get("task_id") if isinstance(item, dict) else item.task_id
        for item in result["task_results"]
    ]
    assert len(task_ids) == 2
    assert len(set(task_ids)) == 2


def test_delegate_emits_completed_event(tmp_path):
    """delegate 的去留决策必须可观测。"""
    from deepresearcher.observability.events import JsonlSink
    from fakes import event_types

    sink = JsonlSink(tmp_path / "events.jsonl")
    supervisor = Supervisor(
        SupervisorLLM(delegate_topics=["一个方向的事实"], complete_args=COMPLETE_ARGS),
        AgentConfig(max_research_rounds=2),
        research_agent=FixedResearcher(),
        event_sink=sink,
    )
    asyncio.run(
        supervisor.run({"query": "研究问题", "clarified_query": "研究问题", "evidences": []})
    )
    types = event_types(tmp_path / "events.jsonl")
    assert "delegate_started" in types
    assert "delegate_completed" in types


class _RecordingDelegateAgent:
    """记录每次派发任务的 run_id，返回可被 ResearcherResult.model_validate 接受的 dict。"""

    def __init__(self) -> None:
        self.task_run_ids: list[str] = []

    async def run(self, task):
        self.task_run_ids.append(task["run_id"])
        return {
            "evidences": [],
            "source_refs": [],
            "task_result": {
                "task_id": task["id"],
                "round": task.get("round", 1),
                "task_index": task.get("sequence", 0),
                "question": task["question"],
                "research_direction": task["question"],
                "execution_status": "completed",
                "coverage_status": "insufficient",
                "evidence_count": 0,
                "source_count": 0,
                "stop_reason": "fallback_complete",
            },
        }


def _delegate_context(run_id: str, current_round: int, agent, *, max_rounds: int = 3):
    from deepresearcher.agents.supervisor.state import SupervisorContext, SupervisorRoundState
    from deepresearcher.observability.scope import AgentScope
    from deepresearcher.state import RunState

    supervisor = Supervisor(
        object(),
        AgentConfig(max_research_rounds=max_rounds, max_tasks_per_round=2),
        research_agent=agent,
    )
    state: RunState = {"query": "q", "clarified_query": "q", "evidences": []}
    round_state = SupervisorRoundState(
        state,
        current_round=current_round,
        active_evidence_limit=10,
    )
    context = SupervisorContext(
        scope=AgentScope(run_id=run_id, agent_name="Supervisor"),
        deps=supervisor._deps,
        round_state=round_state,
    )
    return supervisor, context


def _dispatch(context, topic: str):
    """测试转发：与生产 tools.py 同一入口，参数全部取自 context 的同名字段。"""
    return services.dispatch_research(context.deps, context.round_state, context.scope, topic)


def test_dispatch_research_blocks_when_round_budget_exhausted():
    """轮次超过 max_research_rounds 时返回 blocked，不产生在途登记。"""
    agent = _RecordingDelegateAgent()
    supervisor, context = _delegate_context("run-x", current_round=5, agent=agent, max_rounds=3)
    task, receipt, entry = _dispatch(context, "一个方向")
    assert task is None and entry == {}
    assert receipt["status"] == "blocked"
    assert receipt["reason"] == "round_budget_exhausted"
    assert agent.task_run_ids == []


def test_dispatch_research_allows_repeat_topic_and_scopes_run_id():
    """同 topic 重发各自独立编号与登记；Task run_id 取自各自 context.scope.run_id。"""
    agent_a = _RecordingDelegateAgent()
    sup_a, ctx_a = _delegate_context("run-a", current_round=1, agent=agent_a)
    first_task, first_receipt, first_entry = _dispatch(ctx_a, "重复主题")
    second_task, second_receipt, second_entry = _dispatch(ctx_a, "重复主题")
    assert first_receipt["status"] == "dispatched"
    assert second_receipt["status"] == "dispatched"
    assert first_task["id"] != second_task["id"]
    assert first_task["run_id"] == "run-a" and second_task["run_id"] == "run-a"
    assert first_entry["task_id"] == first_task["id"]
    assert second_entry["payload"]["run_id"] == "run-a"
    assert agent_a.task_run_ids == []  # 受理不执行，方向体在 tools 层拉起

    agent_b = _RecordingDelegateAgent()
    sup_b, ctx_b = _delegate_context("run-b", current_round=1, agent=agent_b)
    task_b, _, entry_b = _dispatch(ctx_b, "重复主题")
    assert task_b["run_id"] == "run-b"
    assert entry_b["payload"]["run_id"] == "run-b"
    assert agent_b.task_run_ids == []  # 无跨 run 泄漏：受理只落本 context 的编号与登记


def _delegate_runtime(context, tool_call_id: str):
    return SimpleNamespace(
        context=context,
        tool_call_id=tool_call_id,
        state={
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": tool_call_id,
                            "name": "ResearchDelegate",
                            "args": {},
                            "type": "tool_call",
                        }
                    ],
                )
            ],
            "task_index_counter": 0,
        },
    )


def _direction_result_with_index(index: int) -> TaskResult:
    return TaskResult(
        task_id=f"task-{index:04d}",
        round=1,
        task_index=index,
        question=f"方向{index}",
        research_direction=f"方向{index}",
        execution_status="completed",
        coverage_status="sufficient",
        evidence_count=0,
        source_count=0,
        stop_reason=TaskStopReason.COMPLETE,
    )


def _inflight_registration(index: int) -> dict:
    return PendingTask(
        task_id=f"task-{index:04d}",
        payload={"id": f"task-{index:04d}", "sequence": index},
        display_title=f"在途{index}",
        round=1,
    ).model_dump()


def test_task_index_continues_after_settled_tasks():
    """新一轮执行（评审打回）不得重用已结算方向的编号，否则编号相同的结论互相覆盖。"""
    round_state = SupervisorRoundState(
        {
            "query": "问题",
            "task_results": [_direction_result_with_index(1), _direction_result_with_index(2)],
        },
        current_round=2,
        active_evidence_limit=6,
    )

    assert round_state.task_index_counter == 2
    assert round_state.allocate_task_index() == 3


def test_task_index_reserves_inflight_tasks():
    """在途登记同样占用编号，编号不能只看已结算结果。"""
    round_state = SupervisorRoundState(
        {
            "query": "问题",
            "task_results": [_direction_result_with_index(1), _direction_result_with_index(2)],
            "pending_tasks": {"task-0005": _inflight_registration(5)},
        },
        current_round=2,
        active_evidence_limit=6,
    )

    assert round_state.task_index_counter == 5
    assert round_state.allocate_task_index() == 6


def test_task_index_for_call_continues_across_rounds():
    """正常派发路径（按调用位置派生）跨轮接续，不回落到 0 基。"""
    round_state = SupervisorRoundState(
        {"query": "问题", "task_results": [_direction_result_with_index(3)]},
        current_round=2,
        active_evidence_limit=6,
    )
    runtime_state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "call-next", "name": "ResearchDelegate", "args": {}, "type": "tool_call"}
                ],
            )
        ],
        "task_index_counter": round_state.initial_channel_state()["task_index_counter"],
    }

    assert round_state.allocate_task_index_for_call(runtime_state, "call-next") == 4


def test_research_delegate_receipt_precedes_direction_completion():
    """派发工具在方向完成前返回受理回执，后台任务体随后才走完。"""
    import time

    async def scenario():
        agent = GatedResearcher()
        supervisor, context = delegate_context_for(agent)
        delegate = next(
            item for item in build_supervisor_tools() if item.name == "ResearchDelegate"
        )
        runtime = _delegate_runtime(context, "call-g1")
        started = time.monotonic()
        command = await delegate.coroutine(
            display_title="慢方向", research_topic="慢方向的事实", runtime=runtime
        )
        assert time.monotonic() - started < 1.0
        message = command.update["messages"][0]
        assert '"status": "dispatched"' in message.content
        entry_add = command.update["pending_tasks"]["add"]
        assert list(entry_add) == ["task-0001"]
        assert command.update["task_index_counter"] == 1
        assert context.task_handles["task-0001"] is not None
        assert agent.entered == [] or agent.entered == ["task-0001"]  # 任务体刚排队，未完成

        agent.gate.set()
        execution = await context.task_handles.pop("task-0001")
        assert execution.task_result.evidence_count == 1

    asyncio.run(scenario())


def test_research_delegate_reentry_creates_single_task():
    """工具重试再入同一调用：幂等门只回受理，不产生第二个方向任务体。"""

    async def scenario():
        agent = GatedResearcher()
        supervisor, context = delegate_context_for(agent)
        delegate = next(
            item for item in build_supervisor_tools() if item.name == "ResearchDelegate"
        )
        runtime = _delegate_runtime(context, "call-retry")
        first = await delegate.coroutine(
            display_title="主题", research_topic="主题的事实", runtime=runtime
        )
        second = await delegate.coroutine(
            display_title="主题", research_topic="主题的事实", runtime=runtime
        )
        assert list(first.update["pending_tasks"]["add"]) == ["task-0001"]
        assert len(context.task_handles) == 1
        assert '"task_id": "task-0001"' in second.update["messages"][0].content
        assert "pending_tasks" not in second.update  # 再入不重复登记
        agent.gate.set()
        await asyncio.gather(*context.task_handles.values())
        context.task_handles.clear()

    asyncio.run(scenario())


def test_direction_task_body_has_no_await_before_agent_run():
    """任务体自 worker_limit 取得后直达 research_agent.run，中途不得插入 await。"""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(services._execute_research_task))
    awaits = [node for node in ast.walk(tree) if isinstance(node, ast.Await)]
    assert len(awaits) == 1  # async with worker_limit 不产生 Await 节点，唯一挂起点即方向执行。
