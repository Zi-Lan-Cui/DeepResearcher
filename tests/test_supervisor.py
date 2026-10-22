import asyncio
import dataclasses
import json
import re
from types import SimpleNamespace
from typing import Annotated, get_args, get_origin, get_type_hints

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import ValidationError

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
    WORKBENCH_INTENT_ADAPTER,
    AspectIssue,
    DirectionStopReason,
    PendingDirection,
    ResearchAgentResult,
    ResearchAspect,
    ResearchDirectionResult,
    ResearchSynthesis,
    StopReason,
    SynthesisFraming,
    UpdateWorkbench,
    WaitDirection,
    WorkbenchUpdateFraming,
)
from fakes import evidence, make_evidence


def test_workbench_arguments_leave_selected_derivation_to_program():
    batch = UpdateWorkbench.model_validate(
        {
            "changes": [
                {
                    "op": "upsert_aspect",
                    "aspect": {
                        "aspect_id": "core",
                        "topic": "核心",
                        "role": "主线",
                        "status": "covered",
                        "conclusion": "已有依据",
                        "evidence_ids": ["e1"],
                    },
                },
                {"op": "framing", "framing": {"thesis": "已有一项事实。"}},
            ]
        }
    )
    assert "selected_evidence_ids" not in str(batch.model_dump())
    intent = WORKBENCH_INTENT_ADAPTER.validate_python(batch.changes[1])
    assert isinstance(intent, WorkbenchUpdateFraming)
    assert intent.framing.answer_goal is None  # 省略字段=不动，由候选合成沿用现稿
    with pytest.raises(ValidationError):  # 拼错的 framing 字段名当场拒绝，不静默丢弃
        SynthesisFraming.model_validate({"unknown_field": 1})


def test_synthesis_derives_stable_evidence_union_from_aspects() -> None:
    synthesis = ResearchSynthesis.model_validate(
        {
            "revision": 1,
            "based_on_working_set_revision": 2,
            "answer_goal": "回答问题",
            "thesis": "综合结论",
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
        "thesis": "综合结论",
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
        ResearchAspect(
            **base, status="uncovered", evidence_ids=["e1"], gaps=[AspectIssue(text="缺来源")]
        )
    with pytest.raises(ValueError, match="必须登记至少一条 gap"):
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
            "thesis": "综合结论",
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
    from deepresearcher.agents.supervisor.state import workbench_view as working_set_snapshot

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
            self.plan: list = []
            self.provider_exhausted = False
            self.research_synthesis = None
            self.max_research_rounds = 1
            self.current_round = 1
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
    UpdateWorkbench 建稿 → Complete;派发回执本身不含证据,不触发修订分支。

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
                and message.name == "UpdateWorkbench"
                and '"status": "applied"' in str(message.content)
                and '"synthesis_revision": 0' not in str(message.content)
            ),
            None,
        )
        if self.ready and revised is not None:
            return AIMessage(content="已建立可部分交付的综合稿，但尚未达到完整标准。")
        if self.complete_args is not None and revised is not None:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "ResearchComplete",
                        "args": {"reason": self.complete_args["reason"]},
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
            has_wait_receipt or not self.delegate_topics
        ):
            evidence_ids = list(dict.fromkeys(re.findall(r'"evidence_id":\s*"([^"]+)"', history)))
            if evidence_ids:
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
                                            "answer_goal": "回答测试问题",
                                            "thesis": "已有 Evidence 支撑核心结论。",
                                            "decision_rationale": (
                                                "已建立最小证据链。"
                                                if self.ready
                                                else self.complete_args["reason"]
                                            ),
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
                                            "evidence_ids": evidence_ids,
                                        },
                                    },
                                ]
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
                        "args": {"reason": "fake 收尾理由。"},
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


_COMPLETE_ARGS = {"reason": "两个方向均有直接来源。"}


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
    assert result["supervisor"].status == "completed"
    assert result["supervisor_next"] == "writer"
    context = [str(message.content) for message in result["supervisor_messages"]]
    assert any("研究委托" in message for message in context)
    # 派发批回执只表受理；方向报告经 WaitDirection 收据送达，含结论与该方向带回的 Evidence claim。
    assert len([message for message in context if '"status": "dispatched"' in message]) == 2
    arrivals = [
        message
        for message in context
        if '"status": "completed"' in message and '"directions":' in message
    ]
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

    # complete_args=None 使 fake 每轮重发同一批方向：产品不做程序化去重，重复
    # 方向每次都真实执行。fake 不会主动收尾，最终由模型调用次数上限终止循环。
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

    # 重复方向每次都真实执行，说明派发层没有去重；未判充分，走部分报告。
    assert len(result["task_results"]) > 1
    assert result["supervisor"].status == "incomplete"
    # 终态归属真正的终止者：模型调用上限优先于轮次预算。
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
    # 拒绝不再是终态：回执留环，模型每一回合都能重提，直到真实终止者出现。
    assert len(complete_receipts) >= 2
    assert '"status": "rejected"' in str(complete_receipts[0].content)
    assert result["supervisor"].status == "incomplete"
    assert result["supervisor_next"] == "render_final_report"
    # 终态归属模型调用上限，而非"ResearchComplete 拒绝"这类协议措辞。
    assert "模型调用预算已耗尽" in result["writer"].feedback


def test_supervisor_stale_complete_rejection_recovers_inside_loop():
    """被拒的 ResearchComplete 留在环内恢复:草稿未反映新桌面被拒,UpdateWorkbench 改对后再提即成立。"""

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
                # wait 已把工作集推进到 revision=1,现稿仍基于 0——故意裸提冻结。
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


def _workbench_tool_messages(result: dict) -> list[ToolMessage]:
    return [
        message
        for message in result["supervisor_messages"]
        if isinstance(message, ToolMessage) and message.name == "UpdateWorkbench"
    ]


def test_freeze_without_draft_loops_to_model_call_ceiling():
    """无草稿冻结被拒且留环恢复；模型反复不成时，终态归属真实终止者=模型调用上限。

    键级协议下"过期版本"场景消失(时间戳程序盖章),留存的等价守卫是:
    协议拒绝不得写进 coverage_gaps,循环靠调用上限结束。
    """

    class NoDraftFreezeLLM:
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
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "ResearchComplete",
                        "args": {"reason": "坚持在无综合稿时冻结"},
                        "id": "call_complete",
                    }
                ],
            )

    supervisor = ResearchSupervisor(
        NoDraftFreezeLLM(),
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

    complete_receipts = [
        message
        for message in result["supervisor_messages"]
        if isinstance(message, ToolMessage) and message.name == "ResearchComplete"
    ]
    assert len(complete_receipts) >= 2  # 反复坚持，每次都被拒
    assert all('"status": "rejected"' in str(message.content) for message in complete_receipts)
    assert result["research_synthesis"] is None
    assert result["working_set_revision"] == 1
    assert "模型调用预算已耗尽" in result["writer"].feedback
    # 协议拒绝不得作为研究缺口写进用户报告的 coverage_gaps 清单。
    assert not any("ResearchComplete" in gap for gap in result["supervisor"].coverage_gaps)


def test_workbench_rejects_evidence_outside_active_working_set():
    """Evidence 在档案但已释放:引用它的批整单点名拒绝——跨键不变式在候选终态上判。"""

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
            if '"slot_removals"' not in history:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "UpdateWorkbench",
                            "args": {"changes": [{"op": "evidence", "remove": evidence_ids[:1]}]},
                            "id": "call_release",
                        }
                    ],
                )
            if '"invalid_evidence_ids"' not in history:
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
                                            "answer_goal": "回答测试问题",
                                            "thesis": "引用已释放 Evidence 的综合稿。",
                                            "decision_rationale": "故意引用非活跃 Evidence。",
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
                                            "conclusion": "仅有已释放的依据。",
                                            "evidence_ids": evidence_ids[:1],
                                        },
                                    },
                                ]
                            },
                            "id": "call_revise",
                        }
                    ],
                )
            return AIMessage(content="桌面已空且无法成稿,结束本轮。")

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

    batches = _workbench_tool_messages(result)
    assert len(batches) == 2  # 一单释放 applied、一单越权整单拒
    rejected = str(batches[1].content)
    assert '"status": "rejected"' in rejected
    assert '"invalid_evidence_ids"' in rejected
    assert "终态活跃证据" in rejected
    assert result["research_synthesis"] is None
    # 活跃集已空且没有落盘综合稿：不能进入写作。
    assert result["supervisor_next"] == "render_final_report"


def test_supervisor_agent_failure_routes_exception_to_failure_details():
    """agent loop 抛异常:异常文本只进执行诊断与内部事件,不得作为"未闭合缺口"进报告。

    模型调用异常会被 ModelRetryMiddleware 软化成引导文本，到不了 loop 级 except；
    这里整体替换 _agent_loop，模拟图级故障（递归上限、checkpointer 等）。
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

    assert result["supervisor"].status == "incomplete"
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
            "thesis": "已有一项可交付结论。",
            "aspects": [
                {
                    "aspect_id": "core",
                    "topic": "核心结论",
                    "role": "主线",
                    "required": True,
                    "status": "partial",
                    "conclusion": "现有证据可部分回答。",
                    "evidence_ids": [item.evidence_id],
                    "gaps": [{"text": "尚有缺口。", "disclose": True}],
                }
            ],
            "selected_evidence_ids": [item.evidence_id],
            "next_actions": [],
            "decision_rationale": "轮次触顶前的最新状态。",
        },
    }
    loop_state = SupervisorLoopState(
        state,
        current_round=0,
        active_evidence_limit=30,
    )
    loop_state.stop_reason = StopReason.ENDED_WITHOUT_COMPLETION

    result = services.compose_final_update(state, loop_state, supervisor.config)

    assert result.run.phase == "writing"
    assert result.supervisor.generation_mode == "partial"
    assert result.research_synthesis is not None
    assert result.research_synthesis.revision == 1
    assert result.writer_directive is not None
    assert result.writer_directive.report_brief.chapters[0].evidence_ids == [item.evidence_id]
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
                and message.name == "UpdateWorkbench"
                and '"status": "applied"' in str(message.content)
            ),
            None,
        )
        if revised is not None:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "ResearchComplete",
                        "args": {"reason": "轮次耗尽，现有综合稿足以成文。"},
                        "id": "call_done",
                    }
                ],
            )
        if "limit exceeded" in tool_output or "round_budget_exhausted" in tool_output:
            history = "\n".join(str(message.content) for message in messages)
            evidence_ids = list(dict.fromkeys(re.findall(r'"evidence_id":\s*"([^"]+)"', history)))
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
                                        "answer_goal": "回答测试问题",
                                        "thesis": "现有事实足以回答问题。",
                                        "decision_rationale": "轮次耗尽且现有材料足以成文。",
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
                                        "conclusion": "已有直接事实",
                                        "evidence_ids": evidence_ids,
                                    },
                                },
                            ]
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
    assert result["supervisor"].status == "completed"
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
    """模型调用上限触发时，终止原因记为上限，不得记成轮次耗尽。"""

    class ReadLoopLLM:
        def bind_tools(self, _tools, tool_choice="any"):
            return self

        async def ainvoke(self, _messages):
            return AIMessage(
                content="",
                tool_calls=[
                    {"name": "ReadWorkbench", "args": {"reason": "查看"}, "id": "call_read"}
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
    """StopReason 是停止原因的唯一来源:值可回环构造,描述与默认值判定都在成员上。"""
    from deepresearcher.schemas import StopReason
    from deepresearcher.schemas.sections import _STOP_REASON_DESCRIPTIONS

    for reason in StopReason:
        assert StopReason(reason.value) is reason
        assert isinstance(reason, str)  # terminal_reason/JSON 等 str 消费点兼容
    # 描述表全覆盖：漏登记的成员会落到默认句，而默认句对显式结束语义相反。
    assert set(_STOP_REASON_DESCRIPTIONS) == set(StopReason)

    # 校验 allows_partial_report 集合与 compose_final_update 里写明的清单一致：
    assert {r for r in StopReason if r.allows_partial_report} == {
        StopReason.ENDED_WITHOUT_COMPLETION,
        StopReason.GLOBAL_ROUND_BUDGET_EXHAUSTED,
        StopReason.MODEL_CALL_LIMIT_EXCEEDED,
    }
    # 描述文案逐字校验；改文案须同步这里：
    assert (
        StopReason.ENDED_WITHOUT_COMPLETION.description
        == "模型结束研究而未显式收尾，按最新工作台交付部分报告。"
    )
    assert StopReason.COMPLETED.description.startswith("Supervisor 完成研究并冻结综合稿")


def test_stop_reason_rank_is_a_declared_total_order():
    """优先级是声明式的：每个成员有确定权威度，相邻关系固定。"""
    from deepresearcher.schemas import StopReason

    ranks = {reason: reason.rank for reason in StopReason}
    assert len(set(ranks.values())) == len(ranks)  # 无并列，全序确定
    # 现存最弱信号是轮次预算，上限必须强于它。
    assert StopReason.MODEL_CALL_LIMIT_EXCEEDED.rank > StopReason.ENDED_WITHOUT_COMPLETION.rank
    assert min(ranks.values()) == StopReason.ENDED_WITHOUT_COMPLETION.rank
    # 模型的显式收尾决定是最高终态，压过基础设施失败。
    assert StopReason.COMPLETED.rank > StopReason.AGENT_FAILED.rank
    # 全局轮次预算强于单轮，且都低于调用上限。
    assert StopReason.GLOBAL_ROUND_BUDGET_EXHAUSTED.rank > StopReason.ENDED_WITHOUT_COMPLETION.rank
    assert StopReason.MODEL_CALL_LIMIT_EXCEEDED.rank > StopReason.GLOBAL_ROUND_BUDGET_EXHAUSTED.rank
    # 账户级熔断认证介于调用上限与真实异常之间。
    assert StopReason.PROVIDER_EXHAUSTED.rank > StopReason.MODEL_CALL_LIMIT_EXCEEDED.rank
    assert StopReason.AGENT_FAILED.rank > StopReason.PROVIDER_EXHAUSTED.rank


def test_loop_state_set_stop_reason_adopts_by_rank():
    """低权威度不覆盖高权威度;None 直接采纳;更强信号后到则升级覆盖。"""
    from deepresearcher.state import ResearchState

    state: ResearchState = {}
    loop_state = SupervisorLoopState(state, current_round=4, active_evidence_limit=30)
    # current_round 只由构造参数注入，持久化 section 不作为第二个来源。
    assert loop_state.current_round == 4

    assert loop_state.stop_reason is None
    loop_state.set_stop_reason(StopReason.ENDED_WITHOUT_COMPLETION)
    assert loop_state.stop_reason == StopReason.ENDED_WITHOUT_COMPLETION

    # 先记录轮次预算、随后命中调用上限 → 应升级为 MODEL_CALL_LIMIT。
    loop_state.set_stop_reason(StopReason.MODEL_CALL_LIMIT_EXCEEDED)
    assert loop_state.stop_reason == StopReason.MODEL_CALL_LIMIT_EXCEEDED

    # 更弱的信号不得回退覆盖已采纳的更强终止原因。
    loop_state.set_stop_reason(StopReason.ENDED_WITHOUT_COMPLETION)
    assert loop_state.stop_reason == StopReason.MODEL_CALL_LIMIT_EXCEEDED

    # 模型的显式收尾决定压过一切基础设施信号。
    loop_state.set_stop_reason(StopReason.COMPLETED)
    assert loop_state.stop_reason == StopReason.COMPLETED


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
    assert agent_a.task_run_ids == []  # 受理不执行，方向体在 tools 层拉起

    agent_b = _RecordingDelegateAgent()
    sup_b, ctx_b = _delegate_context("run-b", current_round=1, agent=agent_b)
    task_b, _, entry_b = _dispatch(ctx_b, "重复主题")
    assert task_b["run_id"] == "run-b"
    assert entry_b["subtask"]["run_id"] == "run-b"
    assert agent_b.task_run_ids == []  # 无跨 run 泄漏：受理只落本 context 的编号与登记


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


def _direction_result_with_index(index: int) -> ResearchDirectionResult:
    return ResearchDirectionResult(
        task_id=f"task-{index:04d}",
        round=1,
        task_index=index,
        question=f"方向{index}",
        research_direction=f"方向{index}",
        execution_status="completed",
        coverage_status="sufficient",
        evidence_count=0,
        source_count=0,
        stop_reason=DirectionStopReason.COMPLETE,
    )


def _inflight_registration(index: int) -> dict:
    return PendingDirection(
        task_id=f"task-{index:04d}",
        subtask={"id": f"task-{index:04d}", "sequence": index},
        display_title=f"在途{index}",
        round=1,
    ).model_dump()


def test_task_index_continues_after_settled_directions():
    """新一轮执行（评审打回）不得重用已结算方向的编号。

    计数器若写死为 0，第二轮又发 task-0001，按 task_id 后写覆盖会顶掉第一轮结论。
    """
    loop_state = SupervisorLoopState(
        {
            "query": "问题",
            "task_results": [_direction_result_with_index(1), _direction_result_with_index(2)],
        },
        current_round=2,
        active_evidence_limit=6,
    )

    assert loop_state.task_index_counter == 2
    assert loop_state.allocate_task_index() == 3


def test_task_index_reserves_inflight_directions():
    """在途登记尚未结算也占用了编号，只看已结算结果会撞上正在跑的方向。"""
    loop_state = SupervisorLoopState(
        {
            "query": "问题",
            "task_results": [_direction_result_with_index(1), _direction_result_with_index(2)],
            "pending_directions": {"task-0005": _inflight_registration(5)},
        },
        current_round=2,
        active_evidence_limit=6,
    )

    assert loop_state.task_index_counter == 5
    assert loop_state.allocate_task_index() == 6


def test_task_index_for_call_continues_across_rounds():
    """正常派发路径（按调用位置派生）跨轮接续，不回落到 0 基。"""
    loop_state = SupervisorLoopState(
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
        "task_index_counter": loop_state.initial_channel_state()["task_index_counter"],
    }

    assert loop_state.allocate_task_index_for_call(runtime_state, "call-next") == 4


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
        assert agent.entered == [] or agent.entered == ["task-0001"]  # 任务体刚排队，未完成

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
        assert not waiter.done()  # 门未开：唯一合法的挂起点就是 wait

        agent.gate.set()
        reports, delta, _timed_out = await waiter
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


def test_wait_direction_returns_immediately_when_arrivals_pending():
    """实场 run-3234 回归:已有到货、另有方向仍在途时,wait 按契约立即返回波次,
    不得把已到的挂起等最后一路——那正是到达制要消灭的批次屏障。"""

    async def scenario():
        agent = _GatedResearchAgent()
        supervisor, context = _delegate_context_for(agent)
        deps = _capturing_deps(supervisor, [])
        arrived_execution = TaskExecution(
            task_result=ResearchDirectionResult(
                task_id="task-0001",
                round=1,
                task_index=1,
                question="已到方向",
                research_direction="已到方向",
                execution_status="completed",
                coverage_status="sufficient",
                evidence_count=1,
                source_count=1,
                stop_reason="complete",
            ),
            evidences=[],
            selected_evidence_ids=[],
            source_refs=[],
        )
        arrived_task: asyncio.Future = asyncio.get_running_loop().create_future()
        arrived_task.set_result(arrived_execution)
        context.loop_state.pending_directions = {
            "task-0001": _pending_entry("task-0001", "已到", "已到")
        }
        context.direction_tasks["task-0001"] = arrived_task

        slow_entry = _pending_entry("task-0002", "仍在途", "在途")
        context.loop_state.pending_directions["task-0002"] = slow_entry
        slow_task = asyncio.ensure_future(
            services._execute_research_task(deps, slow_entry["subtask"])  # type: ignore[arg-type]
        )
        context.direction_tasks["task-0002"] = slow_task
        await asyncio.sleep(0.05)  # slow 已进 gate 等待：在途而非欠账，不得被清算判 lost

        try:
            reports, _delta, _timed_out = await asyncio.wait_for(
                services.wait_for_directions(deps, context.loop_state, context.direction_tasks),
                timeout=1,
            )
        finally:
            agent.gate.set()  # 放行避免悬挂任务
            await asyncio.wait([slow_task])
        assert [item["task_id"] for item in reports] == ["task-0001"]
        assert "task-0001" not in context.direction_tasks
        assert "task-0002" in context.direction_tasks  # 在途的继续挂后台，不被裹挟

    asyncio.run(scenario())


def test_wait_direction_without_inflight_returns_hint():
    async def scenario():
        supervisor, context = _delegate_context_for(FixedResearchAgent())
        reports, delta, _timed_out = await services.wait_for_directions(
            context.deps, context.loop_state, context.direction_tasks
        )
        assert reports == []
        assert delta == {}

    asyncio.run(scenario())


def test_wait_direction_receipts_distinguish_unpinned_from_empty_table():
    """实场 run-88c4/185e 回归:钉不中与无在途必须是两种回执,不能谎报 no_inflight。"""
    with_pinned_mismatch = services.compose_wait_receipt([], {"task-0001"}, 3)
    assert with_pinned_mismatch["status"] == "unknown_task_ids"
    assert with_pinned_mismatch["inflight"] == ["task-0001"]
    assert "task_ids" in str(with_pinned_mismatch["instruction"])

    empty_table = services.compose_wait_receipt([], set(), 3)
    assert empty_table["status"] == "no_inflight"

    landed = services.compose_wait_receipt([{"task_id": "task-0001"}], {"task-0002"}, 4)
    assert landed["status"] == "completed"
    assert "instruction" not in landed


def test_wait_direction_args_strip_injected_control_characters():
    """模型流式长 id 时夹入 \\r(实场 task\\r-0001 全部钉空),入参净化后才进匹配。"""
    parsed = WaitDirection.model_validate({"task_ids": ["task\r-0001", "task-0002 ", ""]})
    assert parsed.task_ids == ["task-0001", "task-0002"]


def test_wait_direction_settles_lost_entries_before_waiting():
    """通道有登记、进程无任务体:wait 先清算欠账(merged_via=lost),绝不挂死等幽灵。"""

    async def scenario():
        events: list[tuple[str, dict]] = []
        supervisor, context = _delegate_context_for(FixedResearchAgent())
        deps = _capturing_deps(supervisor, events)
        context.loop_state.pending_directions = {
            "task-0001": _pending_entry("task-0001", "丢失方向", "丢失")
        }
        reports, delta, _timed_out = await asyncio.wait_for(
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
    """派发后先以纯文本结束;被 nudge 退回后改调 WaitDirection;收据到手再收尾。

    gate 提供时,在改调 WaitDirection 前放行它——用于测试真在途场景里方向的完成时机。
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
            if self.gate is not None:
                self.gate.set()
            return AIMessage(
                content="",
                tool_calls=[{"name": "WaitDirection", "args": {}, "id": "w1"}],
            )
        return AIMessage(content="我以为可以直接结束。")


class _PlanAndDispatchSameReplyModel:
    """首回合同批：UpdateWorkbench 记计划与 ResearchDelegate 派发在同一条回复里发出，两者都执行。"""

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
            isinstance(message, ToolMessage) and message.name == "WaitDirection"
            for message in messages
        )
        if saw_delegate and saw_workbench and not saw_wait:
            return AIMessage(
                content="",
                tool_calls=[{"name": "WaitDirection", "args": {}, "id": "w1"}],
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
    """实跑 run-bb3b 回归：同批不再被串行校验整批驳回，"先写计划再派活"是一等公民。"""
    supervisor = ResearchSupervisor(
        _PlanAndDispatchSameReplyModel(),
        AgentConfig(max_research_rounds=1, max_subtasks_per_round=1),
        research_agent=FixedResearchAgent(),
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
    """只说"接下来派发"不发调用 → 空手文本退场被拦回,第二次才真派发(实跑 run-bb3b 回归)。"""

    def bind_tools(self, _tools, tool_choice="any"):
        return self

    async def ainvoke(self, messages):
        history = "\n".join(str(message.content) for message in messages)
        if "现在还不是收尾时" not in history:
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


def test_bare_exit_without_draft_or_dispatch_is_nudged(tmp_path):
    """无在途但草稿与派发两手空空时,纯文本不算收尾:nudge 后模型补发派发动作。"""
    from deepresearcher.observability.events import JsonlSink
    from fakes import event_types

    sink = JsonlSink(tmp_path / "events.jsonl")
    supervisor = ResearchSupervisor(
        _NarrateWithoutDispatchModel(),
        AgentConfig(max_research_rounds=1),
        research_agent=FixedResearchAgent(),
        event_sink=sink,
    )
    result = asyncio.run(supervisor.run({"query": "q", "clarified_query": "q", "evidences": []}))

    assert "supervisor_tool_loop_nudged" in event_types(tmp_path / "events.jsonl")
    assert [item.task_id for item in result["task_results"]] == ["task-0001"]
    assert "派发过方向" in str(result["supervisor_messages"][-1].content)


def test_supervisor_nudged_when_model_ends_with_inflight(tmp_path):
    """方向仍未完成时,纯文本结束被 submission_guard 拦回;
    模型改用 WaitDirection 后正常收尾,结果经 wait_tool 吸收而非 drain 回退。"""
    from deepresearcher.observability.events import JsonlSink
    from fakes import event_types

    sink = JsonlSink(tmp_path / "events.jsonl")
    agent = _GatedResearchAgent()
    supervisor = ResearchSupervisor(
        _ExitAttemptThenWaitModel(gate=agent.gate),
        AgentConfig(max_research_rounds=1, max_subtasks_per_round=1),
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
    # 本场内主动等待的注入形态是收据：历史里有 WaitDirection 的 ToolMessage。
    context = [str(m.content) for m in result["supervisor_messages"]]
    assert any("端到端延迟" in message and "evidence" in message for message in context)


def test_completed_direction_injected_at_turn_start_without_wait(tmp_path):
    """方向在回合间隔完成时由接收中间件注入,模型无需 WaitDirection 即可合法文本收尾。"""
    from langchain_core.messages import ToolMessage

    from deepresearcher.observability.events import JsonlSink
    from fakes import read_events

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

    records = read_events(tmp_path / "events.jsonl")
    arrivals = [r for r in records if r["event_type"] == "direction_arrival_merged"]
    assert [r["payload"]["merged_via"] for r in arrivals] == ["model_turn"]
    # 已注入即无在途：文本结束合法，既不被 nudge,也没有 WaitDirection 调用。
    assert "supervisor_tool_loop_nudged" not in [r["event_type"] for r in records]
    assert not any(
        isinstance(m, ToolMessage) and m.name == "WaitDirection"
        for m in result["supervisor_messages"]
    )


def test_receiver_middleware_injects_once_and_never_repeats():
    """中间件单测,脱离 agent 装配:注入已完成并移除登记;第二次调用不注入;wait 只见剩余在途。"""
    from types import SimpleNamespace

    from langchain_core.messages import HumanMessage

    from deepresearcher.agents.middleware import DirectionResultReceiverMiddleware

    async def scenario():
        supervisor, context = _delegate_context_for(FixedResearchAgent())
        runtime = SimpleNamespace(context=context)
        middleware = DirectionResultReceiverMiddleware()
        assert await middleware.abefore_model({}, runtime) is None  # 无在途无登记时不注入

        context.loop_state.pending_directions = {
            "task-0001": _pending_entry("task-0001", "方向一", "完成")
        }
        execution = await services._execute_research_task(
            context.deps, context.loop_state.pending_directions["task-0001"]["subtask"]
        )
        ready = asyncio.ensure_future(_instant_task(execution))
        await ready
        context.direction_tasks["task-0001"] = ready

        update = await middleware.abefore_model({}, runtime)
        assert update is not None
        (message,) = update["messages"]
        assert isinstance(message, HumanMessage)
        assert "研究管理观察（方向结果）" in str(message.content)
        assert update["pending_directions"]["remove"] == ["task-0001"]
        assert "task-0001" not in context.loop_state.pending_directions
        assert context.loop_state.task_results[-1].task_id == "task-0001"

        # 已领取的不再注入。
        assert await middleware.abefore_model({}, runtime) is None
        reports, delta, timed_out = await services.wait_for_directions(
            context.deps, context.loop_state, context.direction_tasks
        )
        assert reports == [] and delta == {} and timed_out is False

    async def _instant_task(result):
        return result

    asyncio.run(scenario())


def test_wait_direction_timeout_reports_timed_out():
    """timeout 到期无一完成:回执返回 timed_out,不误报成 no_inflight 或 unknown_task_ids。"""

    async def scenario():
        supervisor, context = _delegate_context_for(FixedResearchAgent())
        context.loop_state.pending_directions = {
            "task-0001": _pending_entry("task-0001", "慢方向", "未完")
        }
        slow: asyncio.Future = asyncio.get_running_loop().create_future()
        context.direction_tasks["task-0001"] = slow
        try:
            reports, delta, timed_out = await services.wait_for_directions(
                context.deps, context.loop_state, context.direction_tasks, timeout=0.2
            )
            assert reports == [] and delta == {} and timed_out is True
            receipt = services.compose_wait_receipt(reports, {"task-0001"}, 3, timed_out=timed_out)
            assert receipt["status"] == "timed_out"
            assert receipt["inflight"] == ["task-0001"]
            # 不带到期标记的同类无报告回执仍按既有判定返回 unknown_task_ids,两种情形不混淆。
            assert (
                services.compose_wait_receipt([], {"task-0001"}, 3)["status"] == "unknown_task_ids"
            )
        finally:
            slow.cancel()

    asyncio.run(scenario())


def _apply_fixture() -> SupervisorLoopState:
    loop_state = SupervisorLoopState(
        {
            "query": "q",
            "evidences": [make_evidence("e1", "事实一"), make_evidence("e2", "事实二")],
            "active_evidence_ids": ["e1"],
        },
        current_round=1,
        active_evidence_limit=6,
    )
    return loop_state


def test_workbench_repairs_double_encoded_changes_payload():
    """模型把 changes 整体或单条操作二次编码成 JSON 字符串时(实场 run-88c4 实证),
    在 schema 层还原后照常走拒绝/施加管线,不得升级为工具调用框架层报错。"""
    framing_op = {"op": "framing", "framing": {"thesis": "现状。"}}
    whole_array = UpdateWorkbench.model_validate(
        {"changes": json.dumps([framing_op, {"op": "acknowledge"}], ensure_ascii=False)}
    )
    assert all(isinstance(item, dict) for item in whole_array.changes)
    per_element = UpdateWorkbench.model_validate(
        {"changes": [json.dumps(framing_op, ensure_ascii=False), {"op": "acknowledge"}]}
    )
    assert isinstance(per_element.changes[0], dict)

    loop_state = _apply_fixture()
    broken = services.apply_workbench_changes(loop_state, ["not-json"])
    assert broken["status"] == "rejected"  # 无法还原的残串仍进拒绝回执，不触框架层
    assert "op" in str(broken["rejections"][0]["reason"])


def test_workbench_rejects_unknown_op_duplicates_and_malformed_aspect():
    loop_state = _apply_fixture()

    unknown = services.apply_workbench_changes(
        loop_state, [{"op": "reserve_evidence", "value": []}]
    )
    assert unknown["status"] == "rejected"
    assert "framing" in str(unknown["rejections"][0]["reason"])  # 可写清单随行告知

    framing_op = {
        "op": "framing",
        "framing": {
            "answer_goal": "目标",
            "thesis": "现状。",
            "decision_rationale": "理由。",
        },
    }
    duplicated = services.apply_workbench_changes(loop_state, [framing_op, framing_op])
    assert duplicated["status"] == "rejected"
    assert duplicated["rejections"][0]["key"] == "framing"

    # aspect 身份只来自 value:键值双写不再可表达，形状错误由 pydantic 逐字段点名。
    malformed = services.apply_workbench_changes(
        loop_state, [{"op": "upsert_aspect", "aspect": {"aspect_id": "core"}}]
    )
    assert malformed["status"] == "rejected"
    assert all(
        issue["field"].startswith("upsert_aspect.aspect.")
        for issue in malformed["rejections"][0]["issues"]
    )
    assert loop_state.research_synthesis is None  # 拒绝批不落盘


def test_workbench_plan_roundtrip_and_aspect_lifecycle():
    loop_state = _apply_fixture()
    applied = services.apply_workbench_changes(
        loop_state,
        [
            {
                "op": "plan",
                "items": [{"content": "核对来源", "status": "in_progress"}],
            },
            {
                "op": "framing",
                "framing": {
                    "answer_goal": "目标",
                    "thesis": "现状。",
                    "decision_rationale": "理由。",
                },
            },
            {
                "op": "upsert_aspect",
                "aspect": {
                    "aspect_id": "core",
                    "topic": "核心",
                    "role": "主线",
                    "status": "covered",
                    "conclusion": "s",
                    "evidence_ids": ["e1"],
                },
            },
        ],
    )
    assert applied["status"] == "applied"
    assert applied["synthesis_revision"] == 1
    assert loop_state.plan == [{"content": "核对来源", "status": "in_progress"}]
    assert loop_state.research_synthesis.selected_evidence_ids == ["e1"]

    removed = services.apply_workbench_changes(
        loop_state, [{"op": "remove_aspect", "aspect_id": "core"}]
    )
    # 草稿期合法：aspect 删空、framing 仍在，稿子存在但过不了竣工验收。
    assert removed["status"] == "applied"
    assert loop_state.research_synthesis is not None
    assert loop_state.research_synthesis.aspects == []
    rejected_freeze = services.freeze_synthesis(loop_state, "空块冻结")
    assert rejected_freeze["status"] == "rejected"
    assert "aspect" in rejected_freeze["rejection_cause"]

    ghost = services.apply_workbench_changes(
        loop_state, [{"op": "remove_aspect", "aspect_id": "ghost"}]
    )
    assert ghost["status"] == "rejected"
    assert "known_aspect_ids" in str(ghost["rejections"][0])

    # plan 通道随 initial/from 往返。
    seeded = loop_state.initial_channel_state()
    restored = SupervisorLoopState.from_channel_values(
        {"plan": seeded["plan"]},
        research_query="q",
        current_round=1,
        active_evidence_limit=6,
    )
    assert restored.plan == loop_state.plan


def test_synthesis_draft_is_free_form_and_acceptance_deferred_to_freeze():
    """草稿期怎么搭都行:单块 aspect 可建稿;三件齐备与至少一块 aspect 全在冻结验收,
    缺谁点名谁;变更后全空则清回未开稿;旧检查点(overall_summary)照读不误。"""
    loop_state = _apply_fixture()
    seeded = services.apply_workbench_changes(
        loop_state,
        [
            {
                "op": "upsert_aspect",
                "aspect": {
                    "aspect_id": "core",
                    "topic": "核心",
                    "role": "主线",
                    "status": "covered",
                    "conclusion": "s",
                    "evidence_ids": ["e1"],
                },
            }
        ],
    )
    assert seeded["status"] == "applied"  # 无 framing 也能开稿
    assert loop_state.research_synthesis.answer_goal == ""

    rejected = services.freeze_synthesis(loop_state, "急着收尾")
    assert rejected["status"] == "rejected"
    cause = rejected["rejection_cause"]
    assert "answer_goal" in cause and "thesis" in cause and "decision_rationale" in cause

    completed = services.apply_workbench_changes(
        loop_state,
        [
            {
                "op": "framing",
                "framing": {
                    "answer_goal": "目标",
                    "thesis": "这是全文要辩护的那一句。",
                    "decision_rationale": "依据。",
                },
            }
        ],
    )
    assert completed["status"] == "applied"
    assert services.freeze_synthesis(loop_state, "验收通过")["status"] == "accepted"
    brief = services.report_brief_from_synthesis(loop_state.completed_synthesis, max_caveats=6)
    assert brief.thesis == "这是全文要辩护的那一句。"

    cleared = services.apply_workbench_changes(
        loop_state,
        [
            {"op": "remove_aspect", "aspect_id": "core"},
            {"op": "framing", "framing": {}},
        ],
    )
    assert cleared["status"] == "applied"  # 三件非空，稿子还在但再次过不了验收
    assert services.freeze_synthesis(loop_state, "拆过头")["status"] == "rejected"

    legacy = ResearchSynthesis.model_validate(
        {
            "revision": 3,
            "based_on_working_set_revision": 2,
            "answer_goal": "旧稿目标",
            "overall_summary": "旧进度摘要被忽略",
            "thesis": "旧稿没有总纲,验收时补。",
            "aspects": [],
            "open_gaps": [],
            "conflicts": [],
            "decision_rationale": "旧依据。",
        }
    )
    assert legacy.thesis == "旧稿没有总纲,验收时补。"


def test_framing_explicit_empty_string_counts_as_untouched():
    """实场 run-3234 回归:模型显式提交 thesis:"" 曾按 min_length 拒绝,白烧一回合;
    空串是不带内容的意图,视作省略,稿子不动、版本不 bump。"""
    framing = SynthesisFraming.model_validate({"thesis": "", "answer_goal": "   "})
    assert framing.thesis is None and framing.answer_goal is None

    loop_state = _apply_fixture()
    services.apply_workbench_changes(
        loop_state, [{"op": "framing", "framing": {"answer_goal": "目标"}}]
    )
    revision_before = loop_state.research_synthesis.revision
    noop = services.apply_workbench_changes(
        loop_state, [{"op": "framing", "framing": {"thesis": ""}}]
    )
    assert noop["status"] == "applied"
    assert loop_state.research_synthesis.revision == revision_before
    assert loop_state.research_synthesis.thesis == ""


def test_workbench_evidence_is_set_semantic_with_same_batch_slot_release():
    """add/remove 是集合语义:id 不得同时出现在两侧（会退回）；
    先释放后找回,同批腾出的槽位立即可用,回执与落盘只有一个事实源。"""
    loop_state = SupervisorLoopState(
        {
            "query": "q",
            "evidences": [make_evidence("e1", "事实一"), make_evidence("e2", "事实二")],
            "active_evidence_ids": ["e1"],
        },
        current_round=1,
        active_evidence_limit=1,
    )

    conflict = services.apply_workbench_changes(
        loop_state, [{"op": "evidence", "add": ["e2"], "remove": ["e2"]}]
    )
    assert conflict["status"] == "rejected"
    assert conflict["rejections"][0]["conflicting_evidence_ids"] == ["e2"]

    swap = services.apply_workbench_changes(
        loop_state, [{"op": "evidence", "add": ["e2"], "remove": ["e1"]}]
    )
    assert swap["status"] == "applied"
    assert swap["slot_additions"] == ["e2"]
    assert swap["slot_removals"] == ["e1"]
    assert set(loop_state.active_evidence_ids) == {"e2"}
    assert "skipped_restore_evidence_ids" not in swap


def test_workbench_batch_resolves_cross_op_dependencies_by_final_state():
    """同批操作间依赖按终态判定:remove 先于 upsert 折叠,结果与模型提交顺序无关;
    evidence 找回与绑定它的 aspect 同批提交即合法。"""
    loop_state = _apply_fixture()  # e1 活跃、e2 在档案
    built = services.apply_workbench_changes(
        loop_state,
        [
            {
                "op": "framing",
                "framing": {
                    "answer_goal": "目标",
                    "thesis": "现状。",
                    "decision_rationale": "理由。",
                },
            },
            {
                "op": "upsert_aspect",
                "aspect": {
                    "aspect_id": "core",
                    "topic": "核心",
                    "role": "主线",
                    "status": "covered",
                    "conclusion": "s",
                    "evidence_ids": ["e1"],
                },
            },
        ],
    )
    assert built["status"] == "applied"

    restored_and_bound = services.apply_workbench_changes(
        loop_state,
        [
            {"op": "evidence", "add": ["e2"]},
            {
                "op": "upsert_aspect",
                "aspect": {
                    "aspect_id": "core",
                    "topic": "核心",
                    "role": "主线",
                    "status": "covered",
                    "conclusion": "补回 e2 后同批绑定",
                    "evidence_ids": ["e1", "e2"],
                },
            },
        ],
    )
    assert restored_and_bound["status"] == "applied"
    assert "e2" in loop_state.active_evidence_ids
    assert loop_state.research_synthesis.selected_evidence_ids == ["e1", "e2"]

    churn = services.apply_workbench_changes(
        loop_state,
        [
            {"op": "remove_aspect", "aspect_id": "core"},
            {
                "op": "upsert_aspect",
                "aspect": {
                    "aspect_id": "core",
                    "topic": "核心",
                    "role": "主线",
                    "status": "covered",
                    "conclusion": "删后重挂",
                    "evidence_ids": ["e1"],
                },
            },
        ],
    )
    assert churn["status"] == "applied"
    aspects = loop_state.research_synthesis.aspects
    assert [aspect.aspect_id for aspect in aspects] == ["core"]
    assert aspects[0].conclusion == "删后重挂"


def test_workbench_acknowledge_restamps_without_editing_draft():
    loop_state = _apply_fixture()
    built = services.apply_workbench_changes(
        loop_state,
        [
            {
                "op": "framing",
                "framing": {
                    "answer_goal": "目标",
                    "thesis": "现状。",
                    "decision_rationale": "理由。",
                },
            },
            {
                "op": "upsert_aspect",
                "aspect": {
                    "aspect_id": "core",
                    "topic": "核心",
                    "role": "主线",
                    "status": "covered",
                    "conclusion": "s",
                    "evidence_ids": ["e1"],
                },
            },
        ],
    )
    assert built["status"] == "applied"
    frozen_before = loop_state.research_synthesis
    assert frozen_before.based_on_working_set_revision == loop_state.working_set_revision

    loop_state._bump_working_set()  # 模拟：一波到达吸收了
    assert not loop_state.synthesis_is_fresh(frozen_before)
    rejected_freeze = services.freeze_synthesis(loop_state, "提前冻结")
    assert rejected_freeze["status"] == "rejected"

    ack = services.apply_workbench_changes(loop_state, [{"op": "acknowledge"}])
    assert ack["status"] == "applied"
    assert loop_state.research_synthesis.revision == 1  # 只重盖章，稿内容与版本不动
    accepted = services.freeze_synthesis(loop_state, "表态后冻结")
    assert accepted["status"] == "accepted"

    eager = services.apply_workbench_changes(
        loop_state,
        [
            {"op": "acknowledge"},
            {"op": "framing", "framing": {"thesis": "又改。"}},
        ],
    )
    assert eager["status"] == "rejected"  # 与实改同单时 acknowledge 冗余，拒绝引导单发


def test_workbench_view_shape_lock():
    loop_state = _apply_fixture()
    from deepresearcher.agents.supervisor.state import workbench_view

    view = workbench_view(loop_state)
    assert set(view) == {
        "workbench_revision",
        "plan",
        "inflight",
        "directions",
        "active_evidence",
        "active_evidence_count",
        "active_evidence_limit",
        "reserve_evidence",
        "reserve_evidence_count",
        "flags",
        "synthesis_revision",
        "research_synthesis",
    }
    assert view["reserve_evidence"][0]["evidence_id"] == "e2"  # 未激活即 reserve 全卡
    assert [item["evidence_id"] for item in view["active_evidence"]] == ["e1"]
    assert view["flags"] == {"provider_exhausted": False, "rounds_left": 0}


def test_freeze_rejected_while_directions_running():
    supervisor, context = _delegate_context_for(FixedResearchAgent())
    context.loop_state.pending_directions = {
        "task-0001": PendingDirection(
            task_id="task-0001", subtask={"id": "task-0001", "question": "在途"}, round=1
        ).model_dump()
    }
    receipt = services.freeze_synthesis(context.loop_state, "想收口")
    assert receipt["status"] == "rejected"
    assert receipt["directions_running"] == ["task-0001"]
    assert "在途" in receipt["rejection_cause"]


def test_drain_absorbs_late_directions_and_registers_nothing_left():
    """节点收尾：等齐在途并吸收，merged_via=drain 落事件；登记清空。"""
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
    assert len(awaits) == 1  # async with worker_limit 不产生 Await 节点，唯一挂起点即方向执行。


def test_render_outcome_vocabulary_and_user_text_lookup():
    """terminal_reason 三域词汇的出口规则:引擎两域查表成中文,lifecycle 原样。"""
    from deepresearcher.schemas import RenderOutcome, StopReason
    from deepresearcher.schemas.sections import _RENDER_OUTCOME_DESCRIPTIONS, terminal_reason_text

    for outcome in RenderOutcome:
        assert RenderOutcome(outcome.value) is outcome
    assert set(_RENDER_OUTCOME_DESCRIPTIONS) == set(RenderOutcome)

    # 第一域：StopReason 值必须映射成中文描述，裸词不得直接进用户报告。
    assert (
        terminal_reason_text(StopReason.ENDED_WITHOUT_COMPLETION)
        == "模型结束研究而未显式收尾，按最新工作台交付部分报告。"
    )
    # 第二域：RenderOutcome 值。
    assert terminal_reason_text(RenderOutcome.REPORT_RENDERED) == "报告已渲染交付。"
    # 第三域：lifecycle 字符串原样透传(自有包装点)。
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
    return services.apply_workbench_changes(
        loop_state,
        [
            {
                "op": "framing",
                "framing": {
                    "answer_goal": "回答研究问题",
                    "thesis": "搜索供应商账户级不可用,无任何证据。",
                    "decision_rationale": "执行故障收束。",
                },
            },
            {
                "op": "upsert_aspect",
                "aspect": {
                    "aspect_id": "core",
                    "topic": "核心结论",
                    "role": "主线",
                    "required": True,
                    "status": "uncovered",
                    "gaps": [{"text": "无法获取材料。", "disclose": True}],
                },
            },
        ],
    )


def test_freeze_synthesis_rejects_evidenceless_without_certification():
    """零证据综合稿默认被拒绝,拒绝回执必须带可行动原因。"""
    loop_state = _blank_loop_state()
    assert _revise_evidenceless(loop_state)["status"] == "applied"

    receipt = services.freeze_synthesis(loop_state, "请求无证据提交")

    assert receipt["status"] == "rejected"
    assert "Evidence" in str(receipt["rejection_cause"])
    assert loop_state.completed_synthesis is None
    assert loop_state.stop_reason is None


def test_absorb_certifies_provider_exhaustion_then_freeze_accepts_evidenceless():
    """链路:方向结果带熔断标志 → absorb 认证 → ResearchComplete 接受零证据稿并记专用终态。"""
    loop_state = _blank_loop_state()
    loop_state.absorb(_exhausted_direction_execution())
    assert loop_state.provider_exhausted is True

    assert _revise_evidenceless(loop_state)["status"] == "applied"
    receipt = services.freeze_synthesis(loop_state, "熔断认证后无证据收束")

    assert receipt["status"] == "accepted"
    assert loop_state.completed_synthesis is not None
    assert loop_state.stop_reason is StopReason.PROVIDER_EXHAUSTED


def test_compose_floor_uses_certified_execution_fault_not_round_budget():
    """默认值按实际原因记录：熔断认证后记 PROVIDER_EXHAUSTED，未认证仍记 ENDED_WITHOUT_COMPLETION。"""
    unproven = _blank_loop_state()
    services.compose_final_update(_blank_state(), unproven, AgentConfig())
    assert unproven.stop_reason is StopReason.ENDED_WITHOUT_COMPLETION

    certified = _blank_loop_state()
    certified.absorb(_exhausted_direction_execution())
    update = services.compose_final_update(_blank_state(), certified, AgentConfig())

    assert certified.stop_reason is StopReason.PROVIDER_EXHAUSTED
    assert update.run.terminal_reason == "supervisor_provider_exhausted"
    # 零证据进不了正文写作：直送终局渲染，诊断文案取自熔断事实而非轮次预算。
    assert update.supervisor_next == "render_final_report"
    assert "账户级不可用" in update.writer.feedback


# 通道化不变式守卫：覆盖 §5 硬点与批重建合同。


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


def test_only_research_delegate_and_workbench_are_concurrent():
    """§5 静态守卫:除 ResearchDelegate 与 UpdateWorkbench（通道互不相交，可同批）外
    每个工具都必须落在串行名单内。"""
    tool_names = {item.name for item in build_supervisor_tools()}
    assert tool_names - set(SUPERVISOR_SERIAL_TOOLS) == {"ResearchDelegate", "UpdateWorkbench"}


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
    assert loop_state.pending_directions == {}  # 吸收即销账，登记不留残值


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
    source.set_stop_reason(StopReason.COMPLETED)

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
    from deepresearcher.agents.supervisor.state import workbench_view as working_set_snapshot

    entry = PendingDirection(
        task_id="task-0001",
        subtask={"id": "task-0001", "question": "方向"},
        display_title="短题",
        round=2,
    ).model_dump()
    source = SupervisorLoopState({"query": "问题"}, current_round=2, active_evidence_limit=6)
    source.pending_directions = {"task-0001": entry}
    # 在途登记不经入口种子：重放再施加 {} 会抹掉已提交登记，只由派发批 Command 写入。
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
    assert working_set_snapshot(empty)["inflight"] == []  # 形状恒有键，不随有无在途漂移


def test_rebuild_middleware_discards_uncommitted_memory():
    """批重建只认通道:未经 Command 提交的内存改动在下一模型回合被替换掉。"""

    class _Runtime:
        def __init__(self, context):
            self.context = context

    committed = SupervisorLoopState(
        {"query": "q", "evidences": []}, current_round=1, active_evidence_limit=6
    )
    # abefore_model 触 loop_state、两个标量与 deps.config 里的上限，scope 用占位对象即可。
    context = SupervisorLoopContext(
        scope=SimpleNamespace(),
        deps=SimpleNamespace(
            config=SimpleNamespace(supervisor_max_active_evidences=6, max_research_rounds=1)
        ),
        loop_state=committed,
        research_query="q",
        current_round=1,
    )
    # 先固化已提交通道(入口 input 在生产里经 langgraph 施加通道 reducer),再对内存对象
    # 做未 Command 提交的改动：重建只认前者。
    committed_channels = committed.initial_channel_state()
    committed_channels["active_evidence_ids"] = merge_active_set(
        None, committed_channels["active_evidence_ids"]
    )
    context.loop_state.provider_exhausted = True
    context.loop_state.working_set_revision = 99

    asyncio.run(LoopStateRebuildMiddleware().abefore_model(committed_channels, _Runtime(context)))
    assert context.loop_state.provider_exhausted is False
    assert context.loop_state.working_set_revision == 0


def test_report_brief_projection_filters_by_disclose():
    """任务书投影:剥管理字段、空结论以首条 gap 顶上、caveats 只收 disclose=true 并按板块归位。"""
    synthesis = ResearchSynthesis(
        revision=1,
        based_on_working_set_revision=0,
        answer_goal="目标",
        thesis="全文主张。",
        aspects=[
            ResearchAspect(
                aspect_id="a1",
                topic="章一",
                role="主线",
                required=True,
                status="covered",
                conclusion="章一结论。",
                key_points=["要点甲", "要点乙"],
                evidence_ids=["e1"],
                gaps=[
                    AspectIssue(text="原文锚点未补", disclose=False),
                    AspectIssue(text="关键期数据缺失", disclose=True),
                ],
                conflicts=[AspectIssue(text="两来源口径不相容", disclose=True)],
            ),
            ResearchAspect(
                aspect_id="a2",
                topic="章二",
                role="边界",
                required=True,
                status="uncovered",
                gaps=[AspectIssue(text="尚未研究", disclose=True)],
            ),
        ],
        decision_rationale="依据。",
    )

    brief = services.report_brief_from_synthesis(synthesis, max_caveats=6)

    assert [chapter.topic for chapter in brief.chapters] == ["章一", "章二"]
    first = brief.chapters[0]
    assert first.key_points == ["要点甲", "要点乙"]
    assert not hasattr(first, "aspect_id") and not hasattr(first, "status")
    assert brief.chapters[1].conclusion == "尚未研究"
    assert brief.caveats == [
        "缺口|板块「章一」:关键期数据缺失",
        "结论冲突|板块「章一」:两来源口径不相容",
        "缺口|板块「章二」:尚未研究",
    ]


def test_legacy_research_synthesis_shape_migrates():
    """旧 checkpoint 结构迁移：summary/remaining_gap 改名生效，全局 gaps/conflicts 并入首块。"""
    legacy = ResearchSynthesis.model_validate(
        {
            "revision": 2,
            "based_on_working_set_revision": 1,
            "answer_goal": "旧目标",
            "thesis": "旧主张",
            "aspects": [
                {
                    "aspect_id": "core",
                    "topic": "核心",
                    "role": "主线",
                    "status": "partial",
                    "summary": "旧摘要",
                    "evidence_ids": [],
                    "remaining_gap": "旧缺口",
                }
            ],
            "open_gaps": ["全局旧缺口"],
            "conflicts": ["旧冲突"],
            "decision_rationale": "旧依据",
            "next_actions": ["早已退役的键"],
        }
    )

    aspect = legacy.aspects[0]
    assert aspect.conclusion == "旧摘要"
    assert [issue.text for issue in aspect.gaps] == ["旧缺口", "全局旧缺口", "旧冲突"]
    assert all(issue.disclose for issue in aspect.gaps)
