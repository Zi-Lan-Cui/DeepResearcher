import asyncio
import dataclasses
import json
import re
from types import SimpleNamespace
from typing import Annotated, get_args, get_origin, get_type_hints

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from deepresearcher.agents.channels import (
    merge_active_set,
    merge_registry_by_key,
    replace_channel,
    retain_if_absent,
)
from deepresearcher.agents.middleware import LoopStateRebuildMiddleware
from deepresearcher.agents.supervisor import ResearchSupervisor, services
from deepresearcher.agents.supervisor.agent import SUPERVISOR_SERIAL_TOOLS
from deepresearcher.agents.supervisor.state import (
    SupervisorAgentState,
    SupervisorDeps,
    SupervisorLoopContext,
    SupervisorLoopState,
    TaskExecution,
    evidence_card,
    synthesis_snapshot,
)
from deepresearcher.agents.supervisor.tools import build_supervisor_tools
from deepresearcher.config import AgentConfig
from deepresearcher.evidence.models import Evidence
from deepresearcher.schemas import (
    DirectionStopReason,
    PendingDirection,
    ResearchAgentResult,
    ResearchAspect,
    ResearchDirectionResult,
    ResearchSynthesis,
    ReviseResearchSynthesis,
    StopReason,
)
from fakes import evidence, make_evidence


def test_synthesis_arguments_leave_selected_evidence_union_to_handler():
    arguments = ReviseResearchSynthesis.model_validate(
        {
            "expected_revision": 0,
            "expected_working_set_revision": 1,
            "answer_goal": "回答问题",
            "overall_summary": "已有一项事实。",
            "aspects": [
                {
                    "aspect_id": "core",
                    "topic": "核心",
                    "role": "主线",
                    "status": "covered",
                    "summary": "已有依据",
                    "evidence_ids": ["e1"],
                }
            ],
            "decision_rationale": "足以部分交付",
        }
    )

    assert "selected_evidence_ids" not in arguments.model_dump()


def test_synthesis_derives_stable_evidence_union_from_aspects() -> None:
    synthesis = ResearchSynthesis.model_validate(
        {
            "revision": 1,
            "based_on_working_set_revision": 2,
            "answer_goal": "回答问题",
            "overall_summary": "综合结论",
            "aspects": [
                {
                    "aspect_id": "a",
                    "topic": "方面 A",
                    "role": "定义",
                    "status": "covered",
                    "evidence_ids": ["e2", "e1"],
                },
                {
                    "aspect_id": "b",
                    "topic": "方面 B",
                    "role": "对比",
                    "status": "covered",
                    "evidence_ids": ["e1", "e3"],
                },
            ],
            "selected_evidence_ids": ["legacy-extra"],
            "decision_rationale": "已形成证据链",
        }
    )

    assert synthesis.selected_evidence_ids == ["e2", "e1", "e3"]


def test_synthesis_rejects_duplicate_aspect_ids() -> None:
    payload = {
        "revision": 1,
        "based_on_working_set_revision": 1,
        "answer_goal": "回答问题",
        "overall_summary": "综合结论",
        "aspects": [
            {
                "aspect_id": "same",
                "topic": f"方面 {index}",
                "role": "主线",
                "status": "covered",
                "evidence_ids": [f"e{index}"],
            }
            for index in (1, 2)
        ],
        "decision_rationale": "已形成证据链",
    }

    with pytest.raises(ValueError, match="重复 aspect_id"):
        ResearchSynthesis.model_validate(payload)


def test_aspect_grounding_requires_evidence_or_explicit_gap() -> None:
    """状态层只校验内部一致性:covered 必须挂证据,uncovered 不得挂证据,缺口必须写明。"""
    base = {"aspect_id": "a", "topic": "主题", "role": "主线"}
    with pytest.raises(ValueError, match="covered 研究方面必须绑定"):
        ResearchAspect(**base, status="covered", evidence_ids=[])
    with pytest.raises(ValueError, match="uncovered 研究方面不能绑定"):
        ResearchAspect(**base, status="uncovered", evidence_ids=["e1"], remaining_gap="缺来源")
    with pytest.raises(ValueError, match="必须明确 remaining_gap"):
        ResearchAspect(**base, status="partial", evidence_ids=["e1"])


def test_supervisor_views_preserve_metadata_without_exposing_quote() -> None:
    item = evidence("metadata", task_id="r1-1")
    item.source_url = "https://docs.example.com/report"
    item.source_title = "来源标题"
    item.published_at = "2026-09-01"
    card = evidence_card(item)

    assert card["source_title"] == "来源标题"
    assert card["source_domain"] == "docs.example.com"
    assert card["published_at"] == "2026-09-01"
    assert "quote" not in card

    synthesis = ResearchSynthesis.model_validate(
        {
            "revision": 1,
            "based_on_working_set_revision": 1,
            "answer_goal": "回答问题",
            "overall_summary": "综合结论",
            "aspects": [
                {
                    "aspect_id": "core",
                    "topic": "核心",
                    "role": "结论主线",
                    "required": False,
                    "status": "covered",
                    "evidence_ids": [item.evidence_id],
                }
            ],
            "decision_rationale": "决策理由",
        }
    )
    snapshot = synthesis_snapshot(synthesis)

    assert snapshot["synthesis_revision"] == 1
    card = snapshot["research_synthesis"]
    assert card["aspects"][0]["role"] == "结论主线"
    assert card["aspects"][0]["required"] is False
    assert card["decision_rationale"] == "决策理由"


def test_working_set_snapshot_resolves_direction_titles_both_forms() -> None:
    """方向短题从 task_results 回查（BaseModel 与检查点 dict 双形态）；无短题回退长契约截断。"""
    from deepresearcher.agents.supervisor.state import working_set_snapshot

    titled = evidence("titled", task_id="task-0001")
    titled.research_direction = "梳理软件服务一般特征的权威定义与出处边界并逐条标注出处要求"
    dict_titled = evidence("dict", task_id="task-0002")
    dict_titled.research_direction = "长契约文本不该出现在带短题的卡上"
    fallback = evidence("fallback", task_id="task-0004")  # 无 task_result：短契约原样显示
    fallback.research_direction = "短方向"
    long_without_title = evidence("long", task_id="task-0003")
    long_without_title.research_direction = "甲" * 60

    class _SnapshotState:
        def __init__(self) -> None:
            self.evidences = [titled, dict_titled, fallback, long_without_title]
            self.active_evidence_ids = {titled.evidence_id}
            self.active_evidence_limit = 6
            self.working_set_revision = 1
            self.pending_directions: dict = {}
            self.task_results = [
                ResearchDirectionResult(
                    task_id="task-0001",
                    round=1,
                    question=titled.research_direction,
                    research_direction=titled.research_direction,
                    display_title="特征体系",
                    execution_status="completed",
                    coverage_status="sufficient",
                    evidence_count=1,
                    source_count=1,
                    stop_reason=DirectionStopReason.COMPLETE,
                ),
                {"task_id": "task-0002", "display_title": "世代语境"},
            ]

        def active_evidences(self) -> list:
            return [item for item in self.evidences if item.evidence_id in self.active_evidence_ids]

    snapshot = working_set_snapshot(_SnapshotState())

    assert snapshot["active_evidence"][0]["direction_title"] == "特征体系"
    reserve_cards = {card["direction_title"] for card in snapshot["reserve_evidence"]}
    assert "世代语境" in reserve_cards  # dict 恢复形态
    assert "甲" * 29 + "…" in reserve_cards  # 无短题的长契约截断到 30 字符
    assert "短方向" in reserve_cards


class SupervisorLLM:
    """工具调用协议的 Supervisor fake:按消息历史无状态决策,并发安全。

    回合节奏与生产契约一致:派发 → WaitDirection 收到达(收据含 "directions")→
    Revise → Complete;派发回执本身不含证据,不触发修订分支。

    bind_tools 返回的模型直接返回带 tool_calls 的 AIMessage。
    """

    def __init__(self, *, delegate_topics, complete_args=None, ready=False):
        self.delegate_topics = delegate_topics
        self.complete_args = complete_args
        self.ready = ready

    def bind_tools(self, _tools, tool_choice="any"):
        return self

    async def ainvoke(self, messages):
        history = "\n".join(str(message.content) for message in messages)
        revised = next(
            (
                message
                for message in reversed(messages)
                if isinstance(message, ToolMessage)
                and message.name == "ReviseResearchSynthesis"
                and '"status": "accepted"' in str(message.content)
            ),
            None,
        )
        if self.ready and revised is not None:
            return AIMessage(content="已建立可部分交付的综合稿，但尚未达到完整标准。")
        if self.complete_args is not None and revised is not None:
            payload = json.loads(str(revised.content).split("\n", 1)[-1])
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "ResearchComplete",
                        "args": {
                            "synthesis_revision": payload["synthesis_revision"],
                            "reason": self.complete_args["reason"],
                        },
                        "id": "call_complete",
                    }
                ],
            )
        has_wait_receipt = any(
            isinstance(message, ToolMessage) and message.name == "WaitDirection"
            for message in messages
        )
        has_delegate_receipt = any(
            isinstance(message, ToolMessage) and message.name == "ResearchDelegate"
            for message in messages
        )
        if has_delegate_receipt and not has_wait_receipt:
            return AIMessage(
                content="",
                tool_calls=[{"name": "WaitDirection", "args": {}, "id": "call_wait"}],
            )
        if (self.complete_args is not None or self.ready) and (
            '"directions":' in history or not self.delegate_topics
        ):
            evidence_ids = list(dict.fromkeys(re.findall(r'"evidence_id":\s*"([^"]+)"', history)))
            if evidence_ids:
                revisions = [
                    int(item) for item in re.findall(r'"working_set_revision":\s*(\d+)', history)
                ]
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "ReviseResearchSynthesis",
                            "args": {
                                "expected_revision": 0,
                                "expected_working_set_revision": max(revisions, default=0),
                                "answer_goal": "回答测试问题",
                                "overall_summary": "已有 Evidence 支撑核心结论。",
                                "aspects": [
                                    {
                                        "aspect_id": "core",
                                        "topic": "核心结论",
                                        "role": "主线",
                                        "required": True,
                                        "status": "covered",
                                        "summary": "直接回答问题",
                                        "evidence_ids": evidence_ids,
                                        "remaining_gap": "",
                                    }
                                ],
                                "open_gaps": [],
                                "conflicts": [],
                                "next_actions": [],
                                "decision_rationale": (
                                    "已建立最小证据链。"
                                    if self.ready
                                    else self.complete_args["reason"]
                                ),
                            },
                            "id": "call_revise",
                        }
                    ],
                )
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "ResearchComplete",
                        "args": self.complete_args,
                        "id": "call_complete",
                    },
                ],
            )
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "ResearchDelegate",
                    "args": {"display_title": "测试短题", "research_topic": topic},
                    "id": f"call_{index}",
                }
                for index, topic in enumerate(self.delegate_topics)
            ],
        )


class FixedResearchAgent:
    async def run(self, task):
        item = evidence(task["id"], task_id=task["id"])
        return ResearchAgentResult(
            evidences=[item],
            selected_evidence_ids=[item.evidence_id],
            source_refs=[item.source_url],
            task_result=ResearchDirectionResult(
                task_id=task["id"],
                round=1,
                task_index=int(task.get("sequence", 1)),
                question=task["question"],
                research_direction=task["question"],
                execution_status="completed",
                coverage_status="sufficient",
                evidence_count=1,
                source_count=1,
                stop_reason="complete",
            ),
        )


_COMPLETE_ARGS = {
    "reason": "两个方向均有直接来源。",
    "synthesis_revision": 1,
}


def test_supervisor_dispatches_independent_research_agents_concurrently():
    config = AgentConfig(
        max_research_rounds=2,
        max_parallel_workers=2,
        research_agent_max_evidences_per_direction=1,
    )
    supervisor = ResearchSupervisor(
        SupervisorLLM(
            delegate_topics=["方向一的事实", "方向二的事实"],
            complete_args=_COMPLETE_ARGS,
        ),
        config,
        research_agent=FixedResearchAgent(),
    )
    result = asyncio.run(
        supervisor.run({"query": "研究问题", "clarified_query": "研究问题", "evidences": []})
    )

    assert len(result["task_results"]) == 2
    assert len(result["evidences"]) == 2
    assert result["supervisor"].is_sufficient is True
    assert result["supervisor_next"] == "writer"
    context = [str(message.content) for message in result["supervisor_messages"]]
    assert any("研究委托" in message for message in context)
    # 派发批回执只表受理;方向报告经 WaitDirection 收据送达,含结论与该方向带回的 Evidence claim。
    assert len([message for message in context if '"status": "dispatched"' in message]) == 2
    arrivals = [message for message in context if '"directions":' in message]
    assert len(arrivals) == 1
    assert "task-0001" in arrivals[0] and "task-0002" in arrivals[0]
    assert all("直接事实" in message for message in arrivals)


def test_concurrent_delegates_allocate_unique_task_ids_before_absorb():
    """任务序号必须分配即消耗,不能从已完成结果反推 max+1。

    读 max+1 的写法下,第二个并行 delegate 会在第一个 absorb 之前读到同一
    序号,两个任务共享 task-0001,merge_task_results 按 task_id 去重时静默
    丢弃一个方向。交错点用 fake 研究代理入口让出一次事件循环来确定性复现。
    """

    class InterleavedResearchAgent(FixedResearchAgent):
        async def run(self, task, **kwargs):
            await asyncio.sleep(0)  # 放行另一路 delegate 走到序号分配
            return await super().run(task, **kwargs)

    supervisor = ResearchSupervisor(
        SupervisorLLM(
            delegate_topics=["方向A的事实", "方向B的事实"],
            complete_args=_COMPLETE_ARGS,
        ),
        AgentConfig(max_research_rounds=2, max_parallel_workers=2),
        research_agent=InterleavedResearchAgent(),
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
    """delegate 的去/留决策必须可观测（否则规划器连续 turn 无法解释）。"""
    from deepresearcher.observability.events import JsonlSink
    from fakes import event_types

    sink = JsonlSink(tmp_path / "events.jsonl")
    supervisor = ResearchSupervisor(
        SupervisorLLM(delegate_topics=["一个方向的事实"], complete_args=_COMPLETE_ARGS),
        AgentConfig(max_research_rounds=2),
        research_agent=FixedResearchAgent(),
        event_sink=sink,
    )
    asyncio.run(
        supervisor.run({"query": "研究问题", "clarified_query": "研究问题", "evidences": []})
    )
    types = event_types(tmp_path / "events.jsonl")
    assert "delegate_started" in types
    assert "delegate_completed" in types


def test_supervisor_requires_research_agent_at_construction():
    with pytest.raises(ValueError, match="ResearchAgent"):
        ResearchSupervisor(
            SupervisorLLM(delegate_topics=[], complete_args=None),
            AgentConfig(),
            research_agent=None,
        )


def test_supervisor_reports_model_ceiling_when_loops_on_same_topic():
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

    # complete_args=None 使 fake 每轮重发同一批方向:产品不做程序化去重,重复
    # 方向每次都真实执行。fake 不会主动收尾,最终由模型调用次数上限终止循环。
    # 声明式 rank 让更强的 MODEL_CALL_LIMIT_EXCEEDED 压过轮次预算(终态归属回归)。
    supervisor = ResearchSupervisor(
        SupervisorLLM(delegate_topics=["重复方向"], complete_args=None),
        AgentConfig(max_research_rounds=3),
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

    # 重复方向每次都真实执行,说明派发层没有去重;未判充分,走部分报告。
    assert len(result["task_results"]) > 1
    assert result["supervisor"].is_sufficient is False
    # 终态归属真正的终止者：模型调用天花板压过轮次预算。
    assert "模型调用预算已耗尽" in result["writer"].feedback


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

    supervisor = ResearchSupervisor(
        SupervisorLLM(delegate_topics=[], complete_args=_COMPLETE_ARGS),
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
    # 拒绝不再是终态:回执留环,模型每一回合都能重提,直到真实终止者出现。
    assert len(complete_receipts) >= 2
    assert '"status": "rejected"' in str(complete_receipts[0].content)
    assert result["supervisor"].is_sufficient is False
    assert result["supervisor_next"] == "render_final_report"
    # 终态归属模型调用天花板，而非"ResearchComplete 拒绝"这类协议措辞。
    assert "模型调用预算已耗尽" in result["writer"].feedback


def test_supervisor_stale_complete_rejection_recovers_inside_loop():
    """被拒的 ResearchComplete 留在环内恢复:带 revision 的回执后,模型改对综合稿再提即成立。"""

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
                    tool_calls=[{"name": "WaitDirection", "args": {}, "id": "call_wait"}],
                )
            if '"status": "rejected"' not in history:
                # 工作集已由 wait 吸收推进到 revision=1，故意引用过期的综合稿。
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "ResearchComplete",
                            "args": {"synthesis_revision": 1, "reason": "错误使用旧版本"},
                            "id": "call_complete_stale",
                        }
                    ],
                )
            accepted_revise = next(
                (
                    message
                    for message in reversed(messages)
                    if isinstance(message, ToolMessage)
                    and message.name == "ReviseResearchSynthesis"
                    and '"status": "accepted"' in str(message.content)
                ),
                None,
            )
            if accepted_revise is None:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "ReviseResearchSynthesis",
                            "args": {
                                "expected_revision": 1,
                                "expected_working_set_revision": 1,
                                "answer_goal": "回答研究问题",
                                "overall_summary": "新增方向已覆盖核心结论。",
                                "aspects": [
                                    {
                                        "aspect_id": "core",
                                        "topic": "核心结论",
                                        "role": "主线",
                                        "required": True,
                                        "status": "covered",
                                        "summary": "直接回答问题",
                                        "evidence_ids": evidence_ids[:1],
                                        "remaining_gap": "",
                                    }
                                ],
                                "open_gaps": [],
                                "conflicts": [],
                                "next_actions": [],
                                "decision_rationale": "按拒绝回执对齐最新工作集后重提。",
                            },
                            "id": "call_revise_fix",
                        }
                    ],
                )
            payload = json.loads(str(accepted_revise.content).split("\n", 1)[-1])
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "ResearchComplete",
                        "args": {
                            "synthesis_revision": payload["synthesis_revision"],
                            "reason": "修正后完成研究。",
                        },
                        "id": "call_complete_fixed",
                    }
                ],
            )

    supervisor = ResearchSupervisor(
        RejectThenRecoverLLM(),
        AgentConfig(max_research_rounds=1),
        research_agent=FixedResearchAgent(),
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
                    "overall_summary": "尚无新增方向结果。",
                    "aspects": [
                        {
                            "aspect_id": "core",
                            "topic": "核心结论",
                            "role": "主线",
                            "required": True,
                            "status": "uncovered",
                            "remaining_gap": "尚未研究",
                        }
                    ],
                    "selected_evidence_ids": [],
                    "open_gaps": ["尚未研究"],
                    "conflicts": [],
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
    assert len(complete_receipts) == 2  # 先拒后收,同环完成自愈
    assert '"status": "rejected"' in str(complete_receipts[0].content)
    assert '"status": "accepted"' in str(complete_receipts[1].content)
    assert result["working_set_revision"] == 1
    assert result["supervisor"].is_sufficient is True
    assert result["supervisor"].generation_mode == "full"
    assert result["supervisor_next"] == "writer"
    assert result["writer_directive"].report_brief is not None


def _revise_tool_messages(result: dict) -> list[ToolMessage]:
    return [
        message
        for message in result["supervisor_messages"]
        if isinstance(message, ToolMessage) and message.name == "ReviseResearchSynthesis"
    ]


def test_revise_synthesis_rejects_stale_working_set_revision():
    """模型基于过期观察提交综合稿：工具返回 stale，不落盘、不覆盖现有综合稿。"""

    class StaleReviseLLM:
        def bind_tools(self, _tools, tool_choice="any"):
            return self

        async def ainvoke(self, messages):
            history = "\n".join(str(message.content) for message in messages)
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
            evidence_ids = list(dict.fromkeys(re.findall(r'"evidence_id":\s*"([^"]+)"', history)))
            if not evidence_ids:
                return AIMessage(
                    content="",
                    tool_calls=[{"name": "WaitDirection", "args": {}, "id": "call_wait"}],
                )
            if '"status": "stale"' not in history:
                # wait 的 absorb 已把工作集推进到 1；故意仍提交派发前看到的 0。
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "ReviseResearchSynthesis",
                            "args": {
                                "expected_revision": 0,
                                "expected_working_set_revision": 0,
                                "answer_goal": "回答测试问题",
                                "overall_summary": "基于过期观察的综合稿。",
                                "aspects": [
                                    {
                                        "aspect_id": "core",
                                        "topic": "核心结论",
                                        "role": "主线",
                                        "required": True,
                                        "status": "uncovered",
                                        "remaining_gap": "尚未整合。",
                                    }
                                ],
                                "open_gaps": [],
                                "conflicts": [],
                                "next_actions": [],
                                "decision_rationale": "故意使用过期工作集版本。",
                            },
                            "id": "call_revise",
                        }
                    ],
                )
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "ResearchComplete",
                        "args": {"synthesis_revision": 1, "reason": "坚持冻结未落盘的综合稿"},
                        "id": "call_complete",
                    }
                ],
            )

    supervisor = ResearchSupervisor(
        StaleReviseLLM(),
        AgentConfig(max_research_rounds=1),
        research_agent=FixedResearchAgent(),
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

    revised = _revise_tool_messages(result)
    assert len(revised) == 1
    assert '"status": "stale"' in str(revised[0].content)
    assert result["research_synthesis"] is None
    assert result["working_set_revision"] == 1
    # 拒绝回执留环恢复;模型反复不成时,终态归属真实终止者,即模型调用天花板。
    assert "模型调用预算已耗尽" in result["writer"].feedback
    # 协议拒绝不得作为研究缺口写进用户报告的 coverage_gaps 清单。
    assert not any("ResearchComplete" in gap for gap in result["supervisor"].coverage_gaps)


def test_revise_synthesis_rejects_evidence_outside_active_working_set():
    """Evidence 仍在档案但已被释放：引用它被工具层拒绝，活跃性检查不与模型内部校验混层。"""

    class ReleasedReferenceLLM:
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
                            "args": {
                                "display_title": "测试短题",
                                "research_topic": "证据将被释放的方向",
                            },
                            "id": "call_delegate",
                        }
                    ],
                )
            if not evidence_ids:
                return AIMessage(
                    content="",
                    tool_calls=[{"name": "WaitDirection", "args": {}, "id": "call_wait"}],
                )
            if '"released_evidence_ids"' not in history:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "ReleaseEvidence",
                            "args": {
                                "evidence_ids": evidence_ids[:1],
                                "reason": "释放以测试引用边界",
                            },
                            "id": "call_release",
                        }
                    ],
                )
            if '"invalid_evidence_ids"' not in history:
                revisions = [
                    int(item) for item in re.findall(r'"working_set_revision":\s*(\d+)', history)
                ]
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "ReviseResearchSynthesis",
                            "args": {
                                "expected_revision": 0,
                                "expected_working_set_revision": max(revisions, default=0),
                                "answer_goal": "回答测试问题",
                                "overall_summary": "引用已释放 Evidence 的综合稿。",
                                "aspects": [
                                    {
                                        "aspect_id": "core",
                                        "topic": "核心结论",
                                        "role": "主线",
                                        "required": True,
                                        "status": "covered",
                                        "summary": "仅有已释放的依据。",
                                        "evidence_ids": evidence_ids[:1],
                                    }
                                ],
                                "open_gaps": [],
                                "conflicts": [],
                                "next_actions": [],
                                "decision_rationale": "故意引用非活跃 Evidence。",
                            },
                            "id": "call_revise",
                        }
                    ],
                )
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "ResearchComplete",
                        "args": {"synthesis_revision": 1, "reason": "坚持终止研究"},
                        "id": "call_complete",
                    }
                ],
            )

    supervisor = ResearchSupervisor(
        ReleasedReferenceLLM(),
        AgentConfig(max_research_rounds=1),
        research_agent=FixedResearchAgent(),
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

    revised = _revise_tool_messages(result)
    assert len(revised) == 1
    content = str(revised[0].content)
    assert '"status": "rejected"' in content
    assert "研究综合稿只能引用当前活跃" in content
    assert result["research_synthesis"] is None
    # 活跃集已空且没有落盘综合稿：不能进入写作。
    assert result["supervisor_next"] == "render_final_report"


def test_supervisor_agent_failure_routes_exception_to_failure_details():
    """agent loop 抛异常:异常文本只进执行诊断与内部事件,不得作为"未闭合缺口"进报告。

    模型调用异常会被 ModelRetryMiddleware 软化成引导文本，到不了 loop 级 except；
    这里整体替换 _agent_loop，模拟图级故障（递归天花板、checkpointer 等）。
    """

    class ExplodingLoop:
        async def ainvoke(self, _input, **_kwargs):
            raise RuntimeError("模拟基础设施异常 secret-detail")

    supervisor = ResearchSupervisor(
        SupervisorLLM(delegate_topics=[], complete_args=None),
        AgentConfig(),
        research_agent=FixedResearchAgent(),
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
    # 内容通道（研究缺口）保持干净，执行细节只出现在诊断文案里。
    assert all("secret-detail" not in gap for gap in result["supervisor"].coverage_gaps)
    assert "执行失败" in result["writer"].feedback
    assert "secret-detail" in result["writer"].feedback


def test_supervisor_allows_partial_report_after_research_budget_exhaustion():
    supervisor = ResearchSupervisor(
        SupervisorLLM(delegate_topics=["一个局部方向"], ready=True),
        AgentConfig(max_research_rounds=1),
        research_agent=FixedResearchAgent(),
    )

    result = asyncio.run(
        supervisor.run({"query": "研究问题", "clarified_query": "研究问题", "evidences": []})
    )

    assert result["supervisor"].is_sufficient is False
    assert result["supervisor"].status == "incomplete"
    assert result["supervisor"].generation_mode == "partial"
    assert result["supervisor_next"] == "writer"
    assert result["run"].phase == "writing"
    assert result["writer_directive"].report_brief is not None


def test_supervisor_freezes_latest_fresh_synthesis_when_round_limit_is_reached():
    """轮次触顶时直接使用最新有效综合稿生成 partial 报告。"""
    item = evidence("latest", task_id="r1-1")
    supervisor = ResearchSupervisor(
        object(),
        AgentConfig(),
        research_agent=object(),
    )
    state = {
        "query": "研究问题",
        "clarified_query": "研究问题",
        "evidences": [item],
        "active_evidence_ids": [item.evidence_id],
        "working_set_revision": 1,
        "research_synthesis": {
            "revision": 1,
            "based_on_working_set_revision": 1,
            "answer_goal": "回答研究问题",
            "overall_summary": "已有一项可交付结论。",
            "aspects": [
                {
                    "aspect_id": "core",
                    "topic": "核心结论",
                    "role": "主线",
                    "required": True,
                    "status": "partial",
                    "summary": "现有证据可部分回答。",
                    "evidence_ids": [item.evidence_id],
                    "remaining_gap": "尚有缺口。",
                }
            ],
            "selected_evidence_ids": [item.evidence_id],
            "open_gaps": ["尚有缺口"],
            "conflicts": [],
            "next_actions": [],
            "decision_rationale": "轮次触顶前的最新状态。",
        },
    }
    loop_state = SupervisorLoopState(
        state,
        current_round=0,
        active_evidence_limit=30,
    )
    loop_state.stop_reason = StopReason.ROUND_BUDGET_EXHAUSTED

    result = services.compose_final_update(state, loop_state, supervisor.config)

    assert result.run.phase == "writing"
    assert result.supervisor.generation_mode == "partial"
    assert result.research_synthesis is not None
    assert result.research_synthesis.revision == 1
    assert result.writer_directive is not None
    assert result.writer_directive.report_brief.covered_topics[0].evidence_ids == [item.evidence_id]
    assert result.writer_directive.evidence_ids == [item.evidence_id]


def test_supervisor_review_rejection_can_continue_research_via_tool_loop():
    class ReviewStateAwareLLM(SupervisorLLM):
        """看到审阅回流消息时补派一个方向,随后给出完整决策。"""

        def __init__(self):
            super().__init__(delegate_topics=["补充方向"], complete_args=_COMPLETE_ARGS)
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

    supervisor = ResearchSupervisor(
        ReviewStateAwareLLM(),
        AgentConfig(max_research_rounds=1, max_post_review_recovery_cycles=1),
        research_agent=FixedResearchAgent(),
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

    assert result["supervisor"].is_sufficient is True
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
                        {"name": "ResearchComplete", "args": _COMPLETE_ARGS, "id": "call_complete"},
                    ],
                )
            return await super().ainvoke(messages)

    supervisor = ResearchSupervisor(
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
                        subtask_id="r1-1",
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
                    "overall_summary": "已有事实可以支撑改写。",
                    "aspects": [
                        {
                            "aspect_id": "core",
                            "topic": "核心结论",
                            "role": "主线",
                            "required": True,
                            "status": "covered",
                            "summary": "已有事实",
                            "evidence_ids": ["e1"],
                        }
                    ],
                    "selected_evidence_ids": ["e1"],
                    "open_gaps": [],
                    "conflicts": [],
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

    assert result["supervisor"].is_sufficient is True
    assert result["supervisor_next"] == "writer"
    assert result["task_results"] == []
    assert "审阅回流" in "\n".join(
        str(message.content) for message in result["supervisor_messages"]
    )


def test_supervisor_review_recovery_limit_preserves_last_draft_for_final_render():
    """超过审阅回流上限后不再调用 Agent，交给终检渲染最后一版。"""
    supervisor = ResearchSupervisor(
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


class _OneEvidenceAgent:
    """每个方向返回一条唯一 Evidence 的最小 ResearchAgent。"""

    def __init__(self):
        self.run_count = 0

    async def run(self, task):
        self.run_count += 1
        item = evidence(f"t{self.run_count}", task_id=task["id"])
        return {
            "evidences": [item],
            "selected_evidence_ids": [item.evidence_id],
            "source_refs": [item.source_url],
            "task_result": {
                "task_id": task["id"],
                "round": int(task.get("round", 1)),
                "task_index": int(task.get("sequence", 1)),
                "question": task["question"],
                "research_direction": task["question"],
                "execution_status": "completed",
                "coverage_status": "direct",
                "evidence_count": 1,
                "source_count": 1,
                "stop_reason": "complete",
            },
        }


class _DelegateUntilBlockedLLM:
    """持续派发新方向；看到本轮派发配额被拦截后调用 ResearchComplete 收尾。"""

    def __init__(self):
        self._index = 0

    def bind_tools(self, _tools, tool_choice="any"):
        return self

    async def ainvoke(self, messages):
        tool_output = "\n".join(
            str(message.content) for message in messages if isinstance(message, ToolMessage)
        )
        revised = next(
            (
                message
                for message in reversed(messages)
                if isinstance(message, ToolMessage)
                and message.name == "ReviseResearchSynthesis"
                and '"status": "accepted"' in str(message.content)
            ),
            None,
        )
        if revised is not None:
            payload = json.loads(str(revised.content).split("\n", 1)[-1])
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "ResearchComplete",
                        "args": {
                            "synthesis_revision": payload["synthesis_revision"],
                            "reason": "轮次耗尽，现有综合稿足以成文。",
                        },
                        "id": "call_done",
                    }
                ],
            )
        if "limit exceeded" in tool_output or "round_budget_exhausted" in tool_output:
            history = "\n".join(str(message.content) for message in messages)
            evidence_ids = list(dict.fromkeys(re.findall(r'"evidence_id":\s*"([^"]+)"', history)))
            revisions = [
                int(item) for item in re.findall(r'"working_set_revision":\s*(\d+)', history)
            ]
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "ReviseResearchSynthesis",
                        "args": {
                            "expected_revision": 0,
                            "expected_working_set_revision": max(revisions, default=0),
                            "answer_goal": "回答测试问题",
                            "overall_summary": "现有事实足以回答问题。",
                            "aspects": [
                                {
                                    "aspect_id": "core",
                                    "topic": "核心结论",
                                    "role": "主线",
                                    "required": True,
                                    "status": "covered",
                                    "summary": "已有直接事实",
                                    "evidence_ids": evidence_ids,
                                }
                            ],
                            "open_gaps": [],
                            "conflicts": [],
                            "next_actions": [],
                            "decision_rationale": "轮次耗尽且现有材料足以成文。",
                        },
                        "id": "call_revise",
                    }
                ],
            )
        self._index += 1
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "ResearchDelegate",
                    "args": {
                        "display_title": "测试短题",
                        "research_topic": f"独立方向{self._index}",
                    },
                    "id": f"call_d{self._index}",
                }
            ],
        )


def test_supervisor_round_limit_blocks_delegation_and_model_can_finish():
    """轮次耗尽后 ResearchDelegate 被本地 hard check 拦截，模型仍可决策成文。"""
    agent = _OneEvidenceAgent()
    existing = evidence("pre", task_id="r0-1")
    supervisor = ResearchSupervisor(
        _DelegateUntilBlockedLLM(),
        AgentConfig(max_research_rounds=1, max_parallel_workers=1),
        research_agent=agent,
    )
    result = asyncio.run(
        supervisor.run(
            {
                "query": "研究问题",
                "clarified_query": "研究问题",
                "evidences": [existing],
                "supervisor": {"status": "incomplete", "current_round": 1},
            }
        )
    )

    assert agent.run_count == 0
    assert result["task_results"] == []
    assert "round_budget_exhausted" in "\n".join(
        str(message.content) for message in result["supervisor_messages"]
    )
    assert result["supervisor"].is_sufficient is True
    assert result["supervisor_next"] == "writer"


def test_supervisor_per_round_delegate_quota_blocks_excess_dispatches():
    """ToolCallLimit 只放行配额内的 ResearchDelegate，超额的派发不执行。"""
    agent = _OneEvidenceAgent()
    supervisor = ResearchSupervisor(
        _DelegateUntilBlockedLLM(),
        AgentConfig(
            max_research_rounds=3,
            max_subtasks_per_round=2,
            max_parallel_workers=1,
        ),
        research_agent=agent,
    )
    result = asyncio.run(
        supervisor.run({"query": "研究问题", "clarified_query": "研究问题", "evidences": []})
    )

    assert agent.run_count == 2
    assert len(result["task_results"]) == 2


def test_supervisor_model_call_limit_no_longer_mislabeled_as_round_budget():
    """模型调用天花板触发时,终止原因记为天花板,不得记成轮次耗尽。"""

    class ReadLoopLLM:
        def bind_tools(self, _tools, tool_choice="any"):
            return self

        async def ainvoke(self, _messages):
            return AIMessage(
                content="",
                tool_calls=[
                    {"name": "ReadWorkingSet", "args": {"reason": "查看"}, "id": "call_read"}
                ],
            )

    supervisor = ResearchSupervisor(
        ReadLoopLLM(),
        AgentConfig(max_research_rounds=3, max_subtasks_per_round=1),
        research_agent=_OneEvidenceAgent(),
    )
    result = asyncio.run(
        supervisor.run({"query": "研究问题", "clarified_query": "研究问题", "evidences": []})
    )

    assert result["run"].terminal_reason == "supervisor_model_call_limit_exceeded"
    assert "模型调用预算已耗尽" in result["writer"].feedback


def test_stop_reason_vocabulary_single_source():
    """StopReason 是停止原因的唯一来源:值可回环构造,描述与兜底判定都在成员上。"""
    from deepresearcher.schemas import StopReason
    from deepresearcher.schemas.sections import _STOP_REASON_DESCRIPTIONS

    for reason in StopReason:
        assert StopReason(reason.value) is reason
        assert isinstance(reason, str)  # terminal_reason/JSON 等 str 消费点兼容
    # 描述表全覆盖:漏登记的成员会落到兜底句,而兜底句对 SUFFICIENT 语义相反。
    assert set(_STOP_REASON_DESCRIPTIONS) == set(StopReason)

    # 校验 allows_partial_report 集合与 compose_final_update 里写明的清单一致:
    assert {r for r in StopReason if r.allows_partial_report} == {
        StopReason.ROUND_BUDGET_EXHAUSTED,
        StopReason.GLOBAL_ROUND_BUDGET_EXHAUSTED,
        StopReason.MODEL_CALL_LIMIT_EXCEEDED,
        StopReason.SUBMITTED_WITH_GAPS,
    }
    # 描述文案逐字校验;改文案须同步这里:
    assert StopReason.ROUND_BUDGET_EXHAUSTED.description == "研究轮次预算已耗尽。"
    assert StopReason.SUFFICIENT.description == "Supervisor 确认现有材料足以形成完整研究报告。"


def test_stop_reason_rank_is_a_declared_total_order():
    """优先级是声明式的：每个成员有确定权威度，相邻关系固定。"""
    from deepresearcher.schemas import StopReason

    ranks = {reason: reason.rank for reason in StopReason}
    assert len(set(ranks.values())) == len(ranks)  # 无并列，全序确定
    # 现存最弱信号是轮次预算,天花板必须强于它。
    assert StopReason.MODEL_CALL_LIMIT_EXCEEDED.rank > StopReason.ROUND_BUDGET_EXHAUSTED.rank
    assert min(ranks.values()) == StopReason.ROUND_BUDGET_EXHAUSTED.rank
    # 模型的显式收尾决定是最高终态，压过基础设施失败。
    assert StopReason.SUFFICIENT.rank > StopReason.AGENT_FAILED.rank
    assert StopReason.SUBMITTED_WITH_GAPS.rank > StopReason.AGENT_FAILED.rank
    # 全局轮次预算强于单轮，且都低于天花板命中。
    assert StopReason.GLOBAL_ROUND_BUDGET_EXHAUSTED.rank > StopReason.ROUND_BUDGET_EXHAUSTED.rank
    assert StopReason.MODEL_CALL_LIMIT_EXCEEDED.rank > StopReason.GLOBAL_ROUND_BUDGET_EXHAUSTED.rank
    # 账户级熔断认证介于天花板与真实异常之间。
    assert StopReason.PROVIDER_EXHAUSTED.rank > StopReason.MODEL_CALL_LIMIT_EXCEEDED.rank
    assert StopReason.AGENT_FAILED.rank > StopReason.PROVIDER_EXHAUSTED.rank


def test_loop_state_set_stop_reason_adopts_by_rank():
    """低权威度不覆盖高权威度;None 直接采纳;更强信号后到则升级覆盖。"""
    from deepresearcher.state import ResearchState

    state: ResearchState = {}
    loop_state = SupervisorLoopState(state, current_round=4, active_evidence_limit=30)
    # current_round 只由构造参数注入,持久化 section 不作为第二个来源。
    assert loop_state.current_round == 4

    assert loop_state.stop_reason is None
    loop_state.set_stop_reason(StopReason.ROUND_BUDGET_EXHAUSTED)
    assert loop_state.stop_reason == StopReason.ROUND_BUDGET_EXHAUSTED

    # 先记录轮次预算、随后命中天花板 → 应升级为 MODEL_CALL_LIMIT。
    loop_state.set_stop_reason(StopReason.MODEL_CALL_LIMIT_EXCEEDED)
    assert loop_state.stop_reason == StopReason.MODEL_CALL_LIMIT_EXCEEDED

    # 更弱的信号不得回退覆盖已采纳的更强终止原因。
    loop_state.set_stop_reason(StopReason.ROUND_BUDGET_EXHAUSTED)
    assert loop_state.stop_reason == StopReason.MODEL_CALL_LIMIT_EXCEEDED

    # 模型的显式收尾决定压过一切基础设施信号。
    loop_state.set_stop_reason(StopReason.SUFFICIENT)
    assert loop_state.stop_reason == StopReason.SUFFICIENT


class _RecordingDelegateAgent:
    """记录每次派发任务的 run_id；返回可被 ResearchAgentResult.model_validate 接受的 dict。"""

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
    from deepresearcher.agents.supervisor.state import SupervisorLoopContext, SupervisorLoopState
    from deepresearcher.observability.execution import AgentExecutionScope
    from deepresearcher.state import ResearchState

    supervisor = ResearchSupervisor(
        object(),
        AgentConfig(max_research_rounds=max_rounds, max_subtasks_per_round=2),
        research_agent=agent,
    )
    state: ResearchState = {"query": "q", "clarified_query": "q", "evidences": []}
    loop_state = SupervisorLoopState(
        state,
        current_round=current_round,
        active_evidence_limit=10,
    )
    context = SupervisorLoopContext(
        scope=AgentExecutionScope(run_id=run_id, agent_name="Supervisor"),
        deps=supervisor._deps,
        loop_state=loop_state,
    )
    return supervisor, context


def _dispatch(context, topic: str):
    """测试转发:与生产 tools.py 同一入口,参数全部来自 context 名词字段。"""
    return services.dispatch_research(context.deps, context.loop_state, context.scope, topic)


def test_dispatch_research_blocks_when_round_budget_exhausted():
    """轮次超过 max_research_rounds 时返回 blocked,不产生在途登记。"""
    agent = _RecordingDelegateAgent()
    supervisor, context = _delegate_context("run-x", current_round=5, agent=agent, max_rounds=3)
    task, receipt, entry = _dispatch(context, "一个方向")
    assert task is None and entry == {}
    assert receipt["status"] == "blocked"
    assert receipt["reason"] == "round_budget_exhausted"
    assert agent.task_run_ids == []


def test_dispatch_research_allows_repeat_topic_and_scopes_run_id():
    """同 topic 重发各自独立编号与登记(无程序化去重,防重复靠提示词纪律+预算封顶)；
    SubTask run_id 取自各自 context.scope.run_id，无跨 run 泄漏。"""
    agent_a = _RecordingDelegateAgent()
    sup_a, ctx_a = _delegate_context("run-a", current_round=1, agent=agent_a)
    first_task, first_receipt, first_entry = _dispatch(ctx_a, "重复主题")
    second_task, second_receipt, second_entry = _dispatch(ctx_a, "重复主题")
    assert first_receipt["status"] == "dispatched"
    assert second_receipt["status"] == "dispatched"
    assert first_task["id"] != second_task["id"]
    assert first_task["run_id"] == "run-a" and second_task["run_id"] == "run-a"
    assert first_entry["task_id"] == first_task["id"]
    assert second_entry["subtask"]["run_id"] == "run-a"
    assert agent_a.task_run_ids == []  # 受理不执行,方向体在 tools 层拉起

    agent_b = _RecordingDelegateAgent()
    sup_b, ctx_b = _delegate_context("run-b", current_round=1, agent=agent_b)
    task_b, _, entry_b = _dispatch(ctx_b, "重复主题")
    assert task_b["run_id"] == "run-b"
    assert entry_b["subtask"]["run_id"] == "run-b"
    assert agent_b.task_run_ids == []  # 无跨 run 泄漏:受理只落本 context 的编号与登记


class _GatedResearchAgent(FixedResearchAgent):
    """完成时机受外部 Event 控制的方向假件:验证派发批不等待方向完成。"""

    def __init__(self) -> None:
        self.gate = asyncio.Event()
        self.entered: list[str] = []

    async def run(self, task):
        self.entered.append(task["id"])
        await self.gate.wait()
        return await super().run(task)


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


def _delegate_context_for(agent, run_id: str = "run-g"):
    from deepresearcher.observability.execution import AgentExecutionScope

    config = AgentConfig(max_research_rounds=3, max_parallel_workers=2)
    supervisor = ResearchSupervisor(
        SupervisorLLM(delegate_topics=["方向"], complete_args=None), config, research_agent=agent
    )
    loop_state = SupervisorLoopState(
        {"query": "问题", "clarified_query": "问题", "evidences": []},
        current_round=1,
        active_evidence_limit=6,
    )
    context = SupervisorLoopContext(
        scope=AgentExecutionScope(run_id=run_id, agent_name="Supervisor"),
        deps=supervisor._deps,
        loop_state=loop_state,
    )
    return supervisor, context


def test_research_delegate_receipt_precedes_direction_completion():
    """派发工具在方向完成前返回受理回执;后台任务体随后才走完。"""
    import time

    async def scenario():
        agent = _GatedResearchAgent()
        supervisor, context = _delegate_context_for(agent)
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
        entry_add = command.update["pending_directions"]["add"]
        assert list(entry_add) == ["task-0001"]
        assert command.update["task_index_counter"] == 1
        assert context.direction_tasks["task-0001"] is not None
        assert agent.entered == [] or agent.entered == ["task-0001"]  # 任务体刚排队,未完成

        agent.gate.set()
        execution = await context.direction_tasks.pop("task-0001")
        assert execution.task_result.evidence_count == 1

    asyncio.run(scenario())


def test_research_delegate_reentry_creates_single_task():
    """工具重试再入同一调用:幂等门只回受理,不产生第二个方向任务体。"""

    async def scenario():
        agent = _GatedResearchAgent()
        supervisor, context = _delegate_context_for(agent)
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
        assert list(first.update["pending_directions"]["add"]) == ["task-0001"]
        assert len(context.direction_tasks) == 1
        assert '"task_id": "task-0001"' in second.update["messages"][0].content
        assert "pending_directions" not in second.update  # 再入不重复登记
        agent.gate.set()
        await asyncio.gather(*context.direction_tasks.values())
        context.direction_tasks.clear()

    asyncio.run(scenario())


def _pending_entry(task_id: str, question: str, title: str) -> dict:
    sequence = int(task_id.split("-")[1])
    subtask = {
        "id": task_id,
        "run_id": "run-g",
        "question": question,
        "display_title": title,
        "round": 1,
        "sequence": sequence,
        "worker_id": f"w-{task_id}",
        "worker_index": sequence,
        "parent_task_id": "",
        "operation_id": f"o-{task_id}",
    }
    return PendingDirection(
        task_id=task_id, subtask=subtask, display_title=title, round=1
    ).model_dump()


def _capturing_deps(supervisor, events: list):
    def capture(event_type, payload, **_kwargs):
        events.append((event_type, payload))

    return dataclasses.replace(supervisor._deps, emit=capture)


def test_wait_direction_blocks_then_returns_reports():
    """唯一消费口:无到货即挂起,到货后收据含完整方向报告并销账。"""

    async def scenario():
        events: list[tuple[str, dict]] = []
        agent = _GatedResearchAgent()
        supervisor, context = _delegate_context_for(agent)
        deps = _capturing_deps(supervisor, events)
        entry = _pending_entry("task-0001", "等待方向", "等待")
        context.loop_state.pending_directions = {"task-0001": entry}
        context.direction_tasks["task-0001"] = asyncio.ensure_future(
            services._execute_research_task(deps, entry["subtask"])  # type: ignore[arg-type]
        )

        waiter = asyncio.ensure_future(
            services.wait_for_directions(deps, context.loop_state, context.direction_tasks)
        )
        await asyncio.sleep(0.05)
        assert not waiter.done()  # 门未开:唯一合法的挂起点就是 wait

        agent.gate.set()
        reports, delta = await waiter
        assert [item["task_id"] for item in reports] == ["task-0001"]
        assert reports[0]["evidence_count"] == 1
        assert delta["pending_directions"] == {"add": {}, "remove": ["task-0001"]}
        assert [item.task_id for item in context.loop_state.task_results] == ["task-0001"]
        arrival = next(
            payload for event_type, payload in events if event_type == "direction_arrival_merged"
        )
        assert arrival["merged_via"] == "wait_tool"
        assert context.direction_tasks == {}

    asyncio.run(scenario())


def test_wait_direction_without_inflight_returns_hint():
    async def scenario():
        supervisor, context = _delegate_context_for(FixedResearchAgent())
        reports, delta = await services.wait_for_directions(
            context.deps, context.loop_state, context.direction_tasks
        )
        assert reports == []
        assert delta == {}

    asyncio.run(scenario())


def test_wait_direction_settles_lost_entries_before_waiting():
    """通道有登记、进程无任务体:wait 先清算欠账(merged_via=lost),绝不挂死等幽灵。"""

    async def scenario():
        events: list[tuple[str, dict]] = []
        supervisor, context = _delegate_context_for(FixedResearchAgent())
        deps = _capturing_deps(supervisor, events)
        context.loop_state.pending_directions = {
            "task-0001": _pending_entry("task-0001", "丢失方向", "丢失")
        }
        reports, delta = await asyncio.wait_for(
            services.wait_for_directions(deps, context.loop_state, context.direction_tasks),
            timeout=1,
        )
        assert reports[0]["status"] == "failed"
        assert "丢失" in reports[0]["research_direction"] or reports[0]["failures"]
        assert delta["pending_directions"]["remove"] == ["task-0001"]
        arrival = next(
            payload for event_type, payload in events if event_type == "direction_arrival_merged"
        )
        assert arrival["merged_via"] == "lost"
        assert context.loop_state.task_results[0].execution_status == "failed"

    asyncio.run(scenario())


class _ExitAttemptThenWaitModel:
    """派完先以纯文本逃逸;被 nudge 退回后改调 WaitDirection;收据到手再收尾。"""

    def bind_tools(self, _tools, **_kwargs):
        return self

    async def ainvoke(self, messages):
        saw_delegate = any(
            getattr(m, "name", None) == "ResearchDelegate" and getattr(m, "type", None) == "tool"
            for m in messages
        )
        saw_wait = any(
            getattr(m, "name", None) == "WaitDirection" and getattr(m, "type", None) == "tool"
            for m in messages
        )
        nudged = any("在途方向未返回" in str(getattr(m, "content", "")) for m in messages)
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
        if nudged:
            return AIMessage(
                content="",
                tool_calls=[{"name": "WaitDirection", "args": {}, "id": "w1"}],
            )
        return AIMessage(content="我以为可以直接结束。")


def test_supervisor_nudged_when_model_ends_with_inflight(tmp_path):
    """有欠账的文本结束被 submission_guard 拦回(不新建中间件,复用收尾提醒);
    模型改用 WaitDirection 后正常收尾,结果经 wait_tool 吸收而非 drain 兜底。"""
    from deepresearcher.observability.events import JsonlSink
    from fakes import event_types

    sink = JsonlSink(tmp_path / "events.jsonl")
    supervisor = ResearchSupervisor(
        _ExitAttemptThenWaitModel(),
        AgentConfig(max_research_rounds=1, max_subtasks_per_round=1),
        research_agent=FixedResearchAgent(),
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
    # 收据是到达的唯一注入形态:历史里有 WaitDirection 的 ToolMessage。
    context = [str(m.content) for m in result["supervisor_messages"]]
    assert any("端到端延迟" in message and "evidence" in message for message in context)


def test_freeze_rejected_while_directions_running():
    supervisor, context = _delegate_context_for(FixedResearchAgent())
    context.loop_state.pending_directions = {
        "task-0001": PendingDirection(
            task_id="task-0001", subtask={"id": "task-0001", "question": "在途"}, round=1
        ).model_dump()
    }
    receipt = services.freeze_synthesis(context.loop_state, 0, "想收口")
    assert receipt["status"] == "rejected"
    assert receipt["directions_running"] == ["task-0001"]
    assert "在途" in receipt["rejection_cause"]


def test_drain_absorbs_late_directions_and_registers_nothing_left():
    """节点收口:等齐在途并吸收,merged_via=drain 落事件;登记清空。"""
    from deepresearcher.observability.execution import AgentExecutionScope

    async def scenario():
        events: list[tuple[str, dict]] = []

        def capture(event_type, payload, **_kwargs):
            events.append((event_type, payload))

        agent = FixedResearchAgent()
        config = AgentConfig(max_research_rounds=3, max_parallel_workers=2)
        supervisor = ResearchSupervisor(
            SupervisorLLM(delegate_topics=["x"], complete_args=None),
            config,
            research_agent=agent,
        )
        deps = dataclasses.replace(supervisor._deps, emit=capture)
        loop_state = SupervisorLoopState(
            {"query": "问题", "evidences": []}, current_round=1, active_evidence_limit=6
        )
        context = SupervisorLoopContext(
            scope=AgentExecutionScope(run_id="run-d", agent_name="Supervisor"),
            deps=deps,
            loop_state=loop_state,
        )
        subtask = {
            "id": "task-0007",
            "run_id": "run-d",
            "question": "迟到方向",
            "display_title": "迟到",
            "round": 1,
            "sequence": 7,
            "worker_id": "w7",
            "worker_index": 7,
            "parent_task_id": "",
            "operation_id": "o7",
        }
        loop_state.pending_directions = {
            "task-0007": PendingDirection(
                task_id="task-0007", subtask=dict(subtask), display_title="迟到", round=1
            ).model_dump()
        }
        context.direction_tasks["task-0007"] = asyncio.ensure_future(
            services._execute_research_task(deps, subtask)  # type: ignore[arg-type]
        )

        await supervisor._drain_direction_tasks(context, loop_state)

        assert [item.task_id for item in loop_state.task_results] == ["task-0007"]
        assert loop_state.pending_directions == {}
        assert context.direction_tasks == {}
        arrival = next(
            payload for event_type, payload in events if event_type == "direction_arrival_merged"
        )
        assert arrival["merged_via"] == "drain"

    asyncio.run(scenario())


def test_residual_tasks_cancelled_not_orphaned():
    async def scenario():
        agent = FixedResearchAgent()
        supervisor, context = _delegate_context_for(agent)
        slow = asyncio.ensure_future(asyncio.sleep(30))
        context.direction_tasks["task-9999"] = slow
        await supervisor._stop_residual_tasks(context)
        assert slow.cancelled()
        assert context.direction_tasks == {}

    asyncio.run(scenario())


def test_direction_task_body_has_no_await_before_agent_run():
    """静态守卫:任务体自 worker_limit 取得后直达 research_agent.run,
    中途多一个 await 就会漂移嵌套 checkpoint ns 的调度序后缀。"""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(services._execute_research_task))
    awaits = [node for node in ast.walk(tree) if isinstance(node, ast.Await)]
    assert len(awaits) == 1  # async with worker_limit 不产生 Await 节点,唯一挂起点即方向执行。


def test_render_outcome_vocabulary_and_user_text_lookup():
    """terminal_reason 三域词汇的出口规则:引擎两域查表成中文,lifecycle 原样。"""
    from deepresearcher.schemas import RenderOutcome, StopReason
    from deepresearcher.schemas.sections import _RENDER_OUTCOME_DESCRIPTIONS, terminal_reason_text

    for outcome in RenderOutcome:
        assert RenderOutcome(outcome.value) is outcome
    assert set(_RENDER_OUTCOME_DESCRIPTIONS) == set(RenderOutcome)

    # 第一域:StopReason 值必须映射成中文描述,裸词不得直接进用户报告。
    assert terminal_reason_text(StopReason.ROUND_BUDGET_EXHAUSTED) == "研究轮次预算已耗尽。"
    # 第二域:RenderOutcome 值。
    assert terminal_reason_text(RenderOutcome.REPORT_RENDERED) == "报告已渲染交付。"
    # 第三域:lifecycle 字符串原样透传(自有包装点)。
    assert terminal_reason_text("user_cancelled") == "user_cancelled"
    assert terminal_reason_text("") == "研究未完成。"


def _exhausted_direction_execution() -> TaskExecution:
    """账户级熔断的已认证方向结果:execution_status=failed 仅表达研究未完成。"""
    return TaskExecution(
        task_result=ResearchDirectionResult(
            task_id="task-0001",
            round=1,
            task_index=1,
            question="检索方向",
            research_direction="检索方向",
            execution_status="failed",
            coverage_status="insufficient",
            evidence_count=0,
            source_count=0,
            failures=["tavily: quota_exhausted"],
            remaining_gaps=["搜索账户级不可用，未获取任何材料。"],
            stop_reason=DirectionStopReason.BLOCKED_WITHOUT_EVIDENCE,
            stop_detail="搜索熔断",
            provider_exhausted=True,
        ),
        evidences=[],
        selected_evidence_ids=[],
        source_refs=[],
    )


def _blank_loop_state() -> SupervisorLoopState:
    return SupervisorLoopState(
        {"query": "研究问题", "clarified_query": "研究问题", "evidences": []},
        current_round=1,
        active_evidence_limit=30,
    )


def _blank_state() -> dict:
    return {"query": "研究问题", "clarified_query": "研究问题", "evidences": []}


def _revise_evidenceless(loop_state: SupervisorLoopState) -> dict:
    current = loop_state.research_synthesis
    return services.revise_synthesis(
        loop_state,
        expected_revision=current.revision if current else 0,
        expected_working_set_revision=loop_state.working_set_revision,
        answer_goal="回答研究问题",
        overall_summary="搜索供应商账户级不可用,无任何证据。",
        aspects=[
            ResearchAspect(
                aspect_id="core",
                topic="核心结论",
                role="主线",
                required=True,
                status="uncovered",
                remaining_gap="无法获取材料。",
            )
        ],
        open_gaps=["无法获取材料。"],
        conflicts=[],
        next_actions=[],
        decision_rationale="执行故障收束。",
    )


def test_freeze_synthesis_rejects_evidenceless_without_certification():
    """零证据综合稿默认被拒绝,拒绝回执必须带可行动原因。"""
    loop_state = _blank_loop_state()
    assert _revise_evidenceless(loop_state)["status"] == "accepted"

    receipt = services.freeze_synthesis(loop_state, 1, "请求无证据提交")

    assert receipt["status"] == "rejected"
    assert "Evidence" in str(receipt["rejection_cause"])
    assert loop_state.completed_synthesis is None
    assert loop_state.stop_reason is None


def test_absorb_certifies_provider_exhaustion_then_freeze_accepts_evidenceless():
    """链路:方向结果带熔断标志 → absorb 认证 → ResearchComplete 接受零证据稿并记专用终态。"""
    loop_state = _blank_loop_state()
    loop_state.absorb(_exhausted_direction_execution())
    assert loop_state.provider_exhausted is True

    assert _revise_evidenceless(loop_state)["status"] == "accepted"
    receipt = services.freeze_synthesis(loop_state, 1, "熔断认证后无证据收束")

    assert receipt["status"] == "accepted"
    assert loop_state.completed_synthesis is not None
    assert loop_state.sufficient is False
    assert loop_state.stop_reason is StopReason.PROVIDER_EXHAUSTED


def test_compose_floor_uses_certified_execution_fault_not_round_budget():
    """默认兜底按实际原因记录:熔断认证后记 PROVIDER_EXHAUSTED,未认证仍记 ROUND_BUDGET_EXHAUSTED。"""
    unproven = _blank_loop_state()
    services.compose_final_update(_blank_state(), unproven, AgentConfig())
    assert unproven.stop_reason is StopReason.ROUND_BUDGET_EXHAUSTED

    certified = _blank_loop_state()
    certified.absorb(_exhausted_direction_execution())
    update = services.compose_final_update(_blank_state(), certified, AgentConfig())

    assert certified.stop_reason is StopReason.PROVIDER_EXHAUSTED
    assert update.run.terminal_reason == "supervisor_provider_exhausted"
    # 零证据进不了正文写作:直送终局渲染,诊断文案取自熔断事实而非轮次预算。
    assert update.supervisor_next == "render_final_report"
    assert "账户级不可用" in update.writer.feedback


# 通道化不变式守卫:覆盖 §5 硬点与批重建合同。


def _reducer_of(hint):
    if get_origin(hint) is Annotated:
        metadata = get_args(hint)[1:]
        return next((arg for arg in metadata if callable(arg)), None)
    return None


def _replace_semantics_channels() -> set[str]:
    """last-write / retain-if-absent 语义的通道:并发整字段快照会互相覆盖,禁止并发写。"""
    hints = get_type_hints(SupervisorAgentState, include_extras=True)
    replace_reducers = {replace_channel, retain_if_absent}
    return {name for name, hint in hints.items() if _reducer_of(hint) in replace_reducers}


def test_only_research_delegate_is_concurrent():
    """§5 静态守卫:除 ResearchDelegate 外的每个工具都必须落在串行名单内。"""
    tool_names = {item.name for item in build_supervisor_tools()}
    assert tool_names - set(SUPERVISOR_SERIAL_TOOLS) == {"ResearchDelegate"}


def test_concurrent_delegate_never_snapshots_replace_channels():
    """派发增量与到达吸收增量只能落在合并语义通道,绝不写 replace 语义通道。"""
    loop_state = SupervisorLoopState(
        {"query": "q", "evidences": []}, current_round=1, active_evidence_limit=6
    )
    subtask = {
        "id": "task-0001",
        "run_id": "run-x",
        "question": "方向",
        "display_title": "",
        "round": 1,
        "sequence": 1,
        "worker_id": "w",
        "worker_index": 1,
        "parent_task_id": "",
        "operation_id": "o",
    }
    entry = PendingDirection(task_id="task-0001", subtask=dict(subtask), round=1).model_dump()
    loop_state.pending_directions = {"task-0001": entry}
    deps = SupervisorDeps(
        config=AgentConfig(),
        research_agent=FixedResearchAgent(),
        worker_limit=asyncio.Semaphore(1),
        emit=lambda *_args, **_kwargs: None,
    )
    execution = TaskExecution.failed_for(subtask, 1, "boom")  # type: ignore[arg-type]
    _, delta = services.absorb_arrival(
        deps,
        loop_state,
        subtask,
        execution,
        merged_via="drain",  # type: ignore[arg-type]
    )
    assert set(delta) & _replace_semantics_channels() == set()
    assert delta["pending_directions"] == {"add": {}, "remove": ["task-0001"]}
    assert loop_state.pending_directions == {}  # 吸收即销账,登记不留残值


def test_loop_state_round_trips_through_channels():
    """initial_channel_state → from_channel_values 与源对象逐字段等价:通道是唯一事实源。"""
    source = SupervisorLoopState(
        {"query": "问题", "clarified_query": "问题", "evidences": []},
        current_round=1,
        active_evidence_limit=6,
    )
    source.absorb(
        TaskExecution(
            task_result=ResearchDirectionResult(
                task_id="task-0001",
                round=1,
                task_index=1,
                question="方向",
                research_direction="方向",
                execution_status="completed",
                coverage_status="sufficient",
                evidence_count=1,
                source_count=1,
                stop_reason="complete",
            ),
            evidences=[make_evidence("e1", "事实")],
            selected_evidence_ids=["e1"],
            source_refs=["https://a"],
        )
    )
    source.set_stop_reason(StopReason.SUFFICIENT)

    channels = source.initial_channel_state()
    # 入口 input 在生产里经 langgraph 施加通道 reducer;测试如实模拟这一步。
    channels["active_evidence_ids"] = merge_active_set(None, channels["active_evidence_ids"])
    restored = SupervisorLoopState.from_channel_values(
        channels,
        research_query=source.research_query,
        current_round=source.current_round,
        active_evidence_limit=source.active_evidence_limit,
    )
    assert [item.evidence_id for item in restored.evidences] == ["e1"]
    assert restored.active_evidence_ids == source.active_evidence_ids
    assert restored.source_refs == source.source_refs
    assert [item.task_id for item in restored.task_results] == ["task-0001"]
    assert restored.working_set_revision == source.working_set_revision
    assert restored.stop_reason is source.stop_reason


def test_merge_registry_by_key_applies_add_and_remove_deltas():
    merged = merge_registry_by_key(None, {"add": {"task-0001": {"round": 1}}, "remove": []})
    merged = merge_registry_by_key(merged, {"add": {"task-0002": {"round": 1}}, "remove": []})
    merged = merge_registry_by_key(merged, {"add": {}, "remove": ["task-0001"]})
    assert sorted(merged) == ["task-0002"]


def test_pending_directions_round_trip_and_inflight_snapshot():
    from deepresearcher.agents.supervisor.state import working_set_snapshot

    entry = PendingDirection(
        task_id="task-0001",
        subtask={"id": "task-0001", "question": "方向"},
        display_title="短题",
        round=2,
    ).model_dump()
    source = SupervisorLoopState({"query": "问题"}, current_round=2, active_evidence_limit=6)
    source.pending_directions = {"task-0001": entry}
    # 在途登记不经入口种子:重放再施加 {} 会抹掉已提交登记,只由派发批 Command 写入。
    assert "pending_directions" not in source.initial_channel_state()

    restored = SupervisorLoopState.from_channel_values(
        {"pending_directions": {"task-0001": entry}},
        research_query="问题",
        current_round=2,
        active_evidence_limit=6,
    )
    assert restored.pending_directions == source.pending_directions
    snapshot = working_set_snapshot(restored)
    assert snapshot["inflight"] == [{"task_id": "task-0001", "display_title": "短题", "round": 2}]
    empty = SupervisorLoopState({"query": "问题"}, current_round=1, active_evidence_limit=6)
    assert working_set_snapshot(empty)["inflight"] == []  # 形状恒有键,不随有无在途漂移


def test_rebuild_middleware_discards_uncommitted_memory():
    """批重建只认通道:未经 Command 提交的内存改动在下一模型回合被替换掉。"""

    class _Runtime:
        def __init__(self, context):
            self.context = context

    committed = SupervisorLoopState(
        {"query": "q", "evidences": []}, current_round=1, active_evidence_limit=6
    )
    # abefore_model 触 loop_state、两个标量与 deps.config 里的上限,scope 用占位对象即可。
    context = SupervisorLoopContext(
        scope=SimpleNamespace(),
        deps=SimpleNamespace(config=SimpleNamespace(supervisor_max_active_evidences=6)),
        loop_state=committed,
        research_query="q",
        current_round=1,
    )
    # 先固化已提交通道(入口 input 在生产里经 langgraph 施加通道 reducer),再对内存对象
    # 做未 Command 提交的改动:重建只认前者。
    committed_channels = committed.initial_channel_state()
    committed_channels["active_evidence_ids"] = merge_active_set(
        None, committed_channels["active_evidence_ids"]
    )
    context.loop_state.provider_exhausted = True
    context.loop_state.working_set_revision = 99

    asyncio.run(LoopStateRebuildMiddleware().abefore_model(committed_channels, _Runtime(context)))
    assert context.loop_state.provider_exhausted is False
    assert context.loop_state.working_set_revision == 0
