"""模型收尾被驳回后的恢复路径：重提、改写、在途未清与注入形态。"""

import asyncio
import re

from langchain_core.messages import AIMessage, ToolMessage

from deepresearcher.agents.supervisor import Supervisor, services
from deepresearcher.agents.supervisor.agent import (
    SUBMISSION_REJECTION_MESSAGE,
)
from deepresearcher.config import AgentConfig
from deepresearcher.evidence.models import Evidence
from supervisor_kit import (
    COMPLETE_ARGS,
    FixedResearcher,
    GatedResearcher,
    SupervisorLLM,
    delegate_context_for,
    pending_entry,
)


def test_supervisor_rejects_completion_without_evidence():
    class EmptyAgent:
        async def run(self, task):
            return {
                "evidences": [],
                "source_refs": [],
                "task_result": {
                    "task_id": task["id"],
                    "round": 1,
                    "question": task["question"],
                    "research_direction": task["question"],
                    "execution_status": "completed",
                    "coverage_status": "insufficient",
                    "task_index": 1,
                    "evidence_count": 0,
                    "source_count": 0,
                    "stop_reason": "fallback_complete",
                },
            }

    supervisor = Supervisor(
        SupervisorLLM(delegate_topics=[], complete_args=COMPLETE_ARGS),
        AgentConfig(max_research_rounds=1),
        research_agent=EmptyAgent(),
    )
    result = asyncio.run(
        supervisor.run(
            {
                "query": "研究问题",
                "clarified_query": "研究问题",
                "evidences": [],
            }
        )
    )

    complete_receipts = [
        message
        for message in result["supervisor_messages"]
        if isinstance(message, ToolMessage) and message.name == "ResearchComplete"
    ]
    # 拒绝不是终态：回执留环，模型每一回合都能重提，直到真实终止者出现。
    assert len(complete_receipts) >= 2
    assert '"status": "rejected"' in str(complete_receipts[0].content)
    assert result["supervisor"].status == "incomplete"
    assert result["supervisor_next"] == "render_final_report"
    # 终态归属模型调用上限，而非"ResearchComplete 拒绝"这类协议措辞。
    assert "模型调用预算已耗尽" in result["writer"].feedback


def test_supervisor_stale_complete_rejection_recovers_inside_loop():
    """被拒的 ResearchComplete 留在环内恢复：草稿未反映新工作台时被拒，改对后再提即成立。"""

    class RejectThenRecoverLLM:
        def bind_tools(self, _tools, tool_choice="any"):
            return self

        async def ainvoke(self, messages):
            history = "\n".join(str(message.content) for message in messages)
            evidence_ids = list(dict.fromkeys(re.findall(r'"evidence_id":\s*"([^"]+)"', history)))
            if "research_direction" not in history:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "ResearchDelegate",
                            "args": {"display_title": "测试短题", "research_topic": "新增证据方向"},
                            "id": "call_delegate",
                        }
                    ],
                )
            if not evidence_ids:
                return AIMessage(
                    content="",
                    tool_calls=[{"name": "WaitTask", "args": {}, "id": "call_wait"}],
                )
            if '"status": "rejected"' not in history:
                # wait 已把工作集推进到 revision=1，现稿仍基于 0——故意裸提冻结。
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "ResearchComplete",
                            "args": {"reason": "错误地直接冻结过期草稿"},
                            "id": "call_complete_stale",
                        }
                    ],
                )
            revised = next(
                (
                    message
                    for message in reversed(messages)
                    if isinstance(message, ToolMessage)
                    and message.name == "UpdateWorkbench"
                    and '"status": "applied"' in str(message.content)
                ),
                None,
            )
            if revised is None:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "UpdateWorkbench",
                            "args": {
                                "changes": [
                                    {
                                        "op": "framing",
                                        "framing": {
                                            "thesis": "新增方向已覆盖核心结论。",
                                        },
                                    },
                                    {
                                        "op": "upsert_aspect",
                                        "aspect": {
                                            "aspect_id": "core",
                                            "topic": "核心结论",
                                            "role": "主线",
                                            "required": True,
                                            "status": "covered",
                                            "conclusion": "直接回答问题",
                                            "evidence_ids": evidence_ids[:1],
                                        },
                                    },
                                ]
                            },
                            "id": "call_revise_fix",
                        }
                    ],
                )
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "ResearchComplete",
                        "args": {"reason": "修正后完成研究。"},
                        "id": "call_complete_fixed",
                    }
                ],
            )

    supervisor = Supervisor(
        RejectThenRecoverLLM(),
        AgentConfig(max_research_rounds=1),
        research_agent=FixedResearcher(),
    )
    result = asyncio.run(
        supervisor.run(
            {
                "query": "研究问题",
                "clarified_query": "研究问题",
                "evidences": [],
                "working_set_revision": 0,
                "research_synthesis": {
                    "revision": 1,
                    "based_on_working_set_revision": 0,
                    "answer_goal": "回答研究问题",
                    "thesis": "尚无新增方向结果。",
                    "aspects": [
                        {
                            "aspect_id": "core",
                            "topic": "核心结论",
                            "role": "主线",
                            "required": True,
                            "status": "uncovered",
                            "gaps": [{"text": "尚未研究", "disclose": True}],
                        }
                    ],
                    "selected_evidence_ids": [],
                    "next_actions": ["新增证据方向"],
                    "decision_rationale": "等待方向结果。",
                },
            }
        )
    )

    complete_receipts = [
        message
        for message in result["supervisor_messages"]
        if isinstance(message, ToolMessage) and message.name == "ResearchComplete"
    ]
    assert len(complete_receipts) == 2  # 先拒后收，同环完成自愈
    assert '"status": "rejected"' in str(complete_receipts[0].content)
    assert '"status": "accepted"' in str(complete_receipts[1].content)
    assert result["working_set_revision"] == 1
    assert result["supervisor"].status == "completed"
    assert result["supervisor"].generation_mode == "full"
    assert result["supervisor_next"] == "writer"
    assert result["writer_directive"].report_brief is not None


def test_supervisor_agent_failure_routes_exception_to_failure_details():
    """agent loop 抛异常时，异常文本只进执行诊断与内部事件，不进报告缺口。

    模型调用异常会被 ModelRetryMiddleware 软化成引导文本，到不了 loop 级 except；
    这里整体替换 _agent_loop 以模拟图级故障。
    """

    class ExplodingLoop:
        async def ainvoke(self, _input, **_kwargs):
            raise RuntimeError("模拟基础设施异常 secret-detail")

    supervisor = Supervisor(
        SupervisorLLM(delegate_topics=[], complete_args=None),
        AgentConfig(),
        research_agent=FixedResearcher(),
    )
    supervisor._agent_loop = ExplodingLoop()
    result = asyncio.run(
        supervisor.run(
            {
                "query": "研究问题",
                "clarified_query": "研究问题",
                "evidences": [],
            }
        )
    )

    assert result["run"].terminal_reason == "supervisor_agent_failed"
    assert result["supervisor_next"] == "render_final_report"
    # 执行细节只出现在诊断文案里，不进任何研究内容通道。
    assert "执行失败" in result["writer"].feedback
    assert "secret-detail" in result["writer"].feedback


def test_supervisor_review_rejection_can_continue_research_via_tool_loop():
    class ReviewStateAwareLLM(SupervisorLLM):
        """看到审阅回流消息时补派一个方向,随后给出完整决策。"""

        def __init__(self):
            super().__init__(delegate_topics=["补充方向"], complete_args=COMPLETE_ARGS)
            self._extra_delegated = False

        async def ainvoke(self, messages):
            history = "\n".join(str(message.content) for message in messages)
            if "## 审阅回流" in history and not self._extra_delegated:
                self._extra_delegated = True
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "ResearchDelegate",
                            "args": {"display_title": "测试短题", "research_topic": "补充方向"},
                            "id": "call_extra",
                        },
                    ],
                )
            return await super().ainvoke(messages)

    supervisor = Supervisor(
        ReviewStateAwareLLM(),
        AgentConfig(max_research_rounds=1, max_post_review_recovery_cycles=1),
        research_agent=FixedResearcher(),
    )
    result = asyncio.run(
        supervisor.run(
            {
                "query": "研究问题",
                "clarified_query": "研究问题",
                "evidences": [],
                "review": {
                    "status": "rejected",
                    "attempts": 1,
                    "feedback": "核心结论缺少直接来源。",
                    "gaps": ["缺方向"],
                },
            }
        )
    )

    assert result["supervisor"].status == "completed"
    assert result["supervisor_next"] == "writer"
    assert len(result["task_results"]) == 1
    assert result["task_results"][0].question == "补充方向"


def test_supervisor_review_rejection_can_rewrite_without_extra_research():
    class RewriteAwareLLM(SupervisorLLM):
        def __init__(self):
            super().__init__(delegate_topics=[], complete_args=None)

        async def ainvoke(self, messages):
            history = "\n".join(str(message.content) for message in messages)
            if "## 审阅回流" in history:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "ResearchComplete", "args": COMPLETE_ARGS, "id": "call_complete"},
                    ],
                )
            return await super().ainvoke(messages)

    supervisor = Supervisor(
        RewriteAwareLLM(),
        AgentConfig(max_research_rounds=0, max_post_review_recovery_cycles=1),
        research_agent=object(),  # 不应被调用
    )
    result = asyncio.run(
        supervisor.run(
            {
                "query": "研究问题",
                "clarified_query": "研究问题",
                "evidences": [
                    Evidence(
                        evidence_id="e1",
                        task_id="r1-1",
                        research_direction="方向",
                        claim="已有事实",
                        quote="原文",
                        source_url="https://example.com",
                    )
                ],
                "active_evidence_ids": ["e1"],
                "working_set_revision": 1,
                "research_synthesis": {
                    "revision": 1,
                    "based_on_working_set_revision": 1,
                    "answer_goal": "回答研究问题",
                    "thesis": "已有事实可以支撑改写。",
                    "aspects": [
                        {
                            "aspect_id": "core",
                            "topic": "核心结论",
                            "role": "主线",
                            "required": True,
                            "status": "covered",
                            "conclusion": "已有事实",
                            "evidence_ids": ["e1"],
                        }
                    ],
                    "selected_evidence_ids": ["e1"],
                    "next_actions": [],
                    "decision_rationale": "只需根据审阅意见改写。",
                },
                "review": {
                    "status": "rejected",
                    "attempts": 1,
                    "feedback": "措辞需要收窄。",
                    "gaps": [],
                },
            }
        )
    )

    assert result["supervisor"].status == "completed"
    assert result["supervisor_next"] == "writer"
    assert result["task_results"] == []
    assert "审阅回流" in "\n".join(
        str(message.content) for message in result["supervisor_messages"]
    )


def test_supervisor_review_recovery_limit_preserves_last_draft_for_final_render():
    """达到审阅回流上限后不调用 Agent，交给终检渲染最后一版。"""
    supervisor = Supervisor(
        object(),
        AgentConfig(max_post_review_recovery_cycles=1),
        research_agent=object(),
    )

    result = asyncio.run(
        supervisor.run(
            {
                "query": "研究问题",
                "clarified_query": "研究问题",
                "report_draft": "最后一版草稿",
                "review": {
                    "status": "rejected",
                    "attempts": 2,
                    "feedback": "仍有非致命缺口",
                },
            }
        )
    )

    assert result["run"].phase == "rendering"
    assert result["run"].terminal_reason == "review_recovery_exhausted"
    assert result["supervisor_next"] == "render_final_report"


class _ExitAttemptThenWaitModel:
    """派发后先以纯文本结束，被打回后改调 WaitTask，收据到手再收尾。

    gate 提供时在改调 WaitTask 前放行，用于控制真在途场景的方向完成时机。
    """

    def __init__(self, gate: asyncio.Event | None = None) -> None:
        self.gate = gate

    def bind_tools(self, _tools, **_kwargs):
        return self

    async def ainvoke(self, messages):
        saw_delegate = any(
            getattr(m, "name", None) == "ResearchDelegate" and getattr(m, "type", None) == "tool"
            for m in messages
        )
        saw_wait = any(
            getattr(m, "name", None) == "WaitTask" and getattr(m, "type", None) == "tool"
            for m in messages
        )
        rejected = any("在途任务未返回" in str(getattr(m, "content", "")) for m in messages)
        if saw_wait:
            return AIMessage(content="已到齐，交回。")
        if not saw_delegate:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "ResearchDelegate",
                        "args": {"display_title": "延迟", "research_topic": "端到端延迟"},
                        "id": "d1",
                    }
                ],
            )
        if rejected:
            if self.gate is not None:
                self.gate.set()
            return AIMessage(
                content="",
                tool_calls=[{"name": "WaitTask", "args": {}, "id": "w1"}],
            )
        return AIMessage(content="我以为可以直接结束。")


class _PlanAndDispatchSameReplyModel:
    """首回合同批：UpdateWorkbench 记计划与 ResearchDelegate 派发在同一回复里发出。"""

    def bind_tools(self, _tools, tool_choice="any"):
        return self

    async def ainvoke(self, messages):
        saw_delegate = any(
            isinstance(message, ToolMessage) and message.name == "ResearchDelegate"
            for message in messages
        )
        saw_workbench = any(
            isinstance(message, ToolMessage) and message.name == "UpdateWorkbench"
            for message in messages
        )
        saw_wait = any(
            isinstance(message, ToolMessage) and message.name == "WaitTask" for message in messages
        )
        if saw_delegate and saw_workbench and not saw_wait:
            return AIMessage(
                content="",
                tool_calls=[{"name": "WaitTask", "args": {}, "id": "w1"}],
            )
        if saw_wait:
            return AIMessage(content="结束。")
        if not (saw_delegate and saw_workbench):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "UpdateWorkbench",
                        "args": {
                            "changes": [
                                {
                                    "op": "plan",
                                    "items": [{"content": "收集复利定义", "status": "in_progress"}],
                                }
                            ]
                        },
                        "id": "u1",
                    },
                    {
                        "name": "ResearchDelegate",
                        "args": {"display_title": "定义方向", "research_topic": "复利的定义"},
                        "id": "d1",
                    },
                ],
            )
        return AIMessage(content="结束。")


def test_workbench_and_delegate_may_share_a_reply():
    """同一回复里记计划与派发两个调用都执行，不被串行校验整批驳回。"""
    supervisor = Supervisor(
        _PlanAndDispatchSameReplyModel(),
        AgentConfig(max_research_rounds=1, max_tasks_per_round=1),
        research_agent=FixedResearcher(),
    )
    result = asyncio.run(supervisor.run({"query": "q", "clarified_query": "q", "evidences": []}))

    receipts = [
        message
        for message in result["supervisor_messages"]
        if isinstance(message, ToolMessage)
        and message.name in {"UpdateWorkbench", "ResearchDelegate"}
    ]
    assert len(receipts) == 2  # 两个工具都执行了，而非一条 mixed-batch 驳回
    assert [item.task_id for item in result["task_results"]] == ["task-0001"]


class _NarrateWithoutDispatchModel:
    """首轮只发纯文本不发调用，被打回后第二次才真派发。"""

    def bind_tools(self, _tools, tool_choice="any"):
        return self

    async def ainvoke(self, messages):
        history = "\n".join(str(message.content) for message in messages)
        if SUBMISSION_REJECTION_MESSAGE not in history:
            return AIMessage(content="接下来我将派发一个方向补足缺口。")
        saw_delegate_receipt = any(
            isinstance(message, ToolMessage) and message.name == "ResearchDelegate"
            for message in messages
        )
        if not saw_delegate_receipt:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "ResearchDelegate",
                        "args": {"display_title": "被提醒后", "research_topic": "真实派发"},
                        "id": "d1",
                    }
                ],
            )
        return AIMessage(content="派发过方向,带着结果收尾。")


def test_bare_exit_without_draft_or_dispatch_is_rejected(tmp_path):
    """无在途且草稿与派发都为空时，纯文本不算收尾，打回后模型补发派发动作。"""
    from deepresearcher.observability.events import JsonlSink
    from fakes import event_types

    sink = JsonlSink(tmp_path / "events.jsonl")
    supervisor = Supervisor(
        _NarrateWithoutDispatchModel(),
        AgentConfig(max_research_rounds=1),
        research_agent=FixedResearcher(),
        event_sink=sink,
    )
    result = asyncio.run(supervisor.run({"query": "q", "clarified_query": "q", "evidences": []}))

    assert "supervisor_tool_loop_nudged" in event_types(tmp_path / "events.jsonl")
    assert [item.task_id for item in result["task_results"]] == ["task-0001"]
    assert "派发过方向" in str(result["supervisor_messages"][-1].content)


def test_supervisor_rejects_model_ending_with_inflight(tmp_path):
    """方向仍未完成时，纯文本结束被 submission_guard 拦回；
    模型改用 WaitTask 后正常收尾，结果经 wait_tool 吸收而非 drain 回退。"""
    from deepresearcher.observability.events import JsonlSink
    from fakes import event_types

    sink = JsonlSink(tmp_path / "events.jsonl")
    agent = GatedResearcher()
    supervisor = Supervisor(
        _ExitAttemptThenWaitModel(gate=agent.gate),
        AgentConfig(max_research_rounds=1, max_tasks_per_round=1),
        research_agent=agent,
        event_sink=sink,
    )
    result = asyncio.run(
        supervisor.run({"query": "端到端延迟", "clarified_query": "端到端延迟", "evidences": []})
    )

    types = event_types(tmp_path / "events.jsonl")
    assert "supervisor_tool_loop_nudged" in types
    assert "direction_arrival_merged" in types
    assert [item.task_id for item in result["task_results"]] == ["task-0001"]
    assert result["task_results"][0].execution_status == "completed"
    # 本场内主动等待的注入形态是收据：历史里有 WaitTask 的 ToolMessage。
    context = [str(m.content) for m in result["supervisor_messages"]]
    assert any("端到端延迟" in message and "evidence" in message for message in context)


def test_completed_direction_injected_at_turn_start_without_wait(tmp_path):
    """方向在回合间隔完成时由接收中间件注入，模型无需 WaitTask 即可合法文本收尾。"""
    from langchain_core.messages import ToolMessage

    from deepresearcher.observability.events import JsonlSink
    from fakes import read_events

    sink = JsonlSink(tmp_path / "events.jsonl")
    supervisor = Supervisor(
        _ExitAttemptThenWaitModel(),
        AgentConfig(max_research_rounds=1, max_tasks_per_round=1),
        research_agent=FixedResearcher(),
        event_sink=sink,
    )
    result = asyncio.run(
        supervisor.run({"query": "端到端延迟", "clarified_query": "端到端延迟", "evidences": []})
    )

    records = read_events(tmp_path / "events.jsonl")
    arrivals = [r for r in records if r["event_type"] == "direction_arrival_merged"]
    assert [r["payload"]["merged_via"] for r in arrivals] == ["model_turn"]
    # 已注入即无在途：文本结束合法，既不被驳回，也没有 WaitTask 调用。
    assert "supervisor_tool_loop_nudged" not in [r["event_type"] for r in records]
    assert not any(
        isinstance(m, ToolMessage) and m.name == "WaitTask" for m in result["supervisor_messages"]
    )


def test_receiver_middleware_injects_once_and_never_repeats():
    """中间件单测，脱离 agent 装配：注入已完成并移除登记，第二次调用不注入，wait 只见剩余在途。"""
    from types import SimpleNamespace

    from langchain_core.messages import HumanMessage

    from deepresearcher.agents.supervisor.middleware import (
        DirectionResultReceiverMiddleware,
    )

    async def scenario():
        supervisor, context = delegate_context_for(FixedResearcher())
        runtime = SimpleNamespace(context=context)
        middleware = DirectionResultReceiverMiddleware()
        assert await middleware.abefore_model({}, runtime) is None  # 无在途无登记时不注入

        context.round_state.pending_tasks = {
            "task-0001": pending_entry("task-0001", "方向一", "完成")
        }
        execution = await services._execute_research_task(
            context.deps, context.round_state.pending_tasks["task-0001"]["payload"]
        )
        ready = asyncio.ensure_future(_instant_task(execution))
        await ready
        context.task_handles["task-0001"] = ready

        update = await middleware.abefore_model({}, runtime)
        assert update is not None
        (message,) = update["messages"]
        assert isinstance(message, HumanMessage)
        assert "研究管理观察（任务结果）" in str(message.content)
        assert update["pending_tasks"]["remove"] == ["task-0001"]
        assert "task-0001" not in context.round_state.pending_tasks
        assert context.round_state.task_results[-1].task_id == "task-0001"

        # 已领取的不注入。
        assert await middleware.abefore_model({}, runtime) is None
        reports, delta, timed_out = await services.wait_for_tasks(
            context.deps, context.round_state, context.task_handles
        )
        assert reports == [] and delta == {} and timed_out is False

    async def _instant_task(result):
        return result

    asyncio.run(scenario())
