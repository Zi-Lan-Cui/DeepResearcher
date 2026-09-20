import asyncio
import json
import re

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from deepresearcher.agents.supervisor import ResearchSupervisor, services
from deepresearcher.agents.supervisor.state import (
    SupervisorLoopState,
    evidence_card,
    synthesis_snapshot,
)
from deepresearcher.config import AgentConfig
from deepresearcher.evidence.models import Evidence
from deepresearcher.schemas import (
    ResearchAgentResult,
    ResearchAspect,
    ResearchDirectionResult,
    ResearchSynthesis,
    ReviseResearchSynthesis,
    StopReason,
)
from fakes import evidence


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
    """状态层只校验内部一致性：covered 必须挂证据、uncovered 不得伪装、缺口必须写明。"""
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

    assert snapshot["aspects"][0]["role"] == "结论主线"
    assert snapshot["aspects"][0]["required"] is False
    assert snapshot["decision_rationale"] == "决策理由"


class SupervisorLLM:
    """工具调用协议的 Supervisor fake:按消息历史无状态决策,并发安全。

    - 历史中没有研究工具结果(以 ToolMessage 载荷标记 "research_direction")时:派发 ResearchDelegate;
    - 已有研究工具结果时:按 complete 参数决定输出 ResearchComplete(或继续派发)。

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
        if (self.complete_args is not None or self.ready) and (
            "research_direction" in history or not self.delegate_topics
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
                    "args": {"research_topic": topic},
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
    assert any("研究管理观察" in message for message in context)
    # 工具结果以 JSON 载荷注入历史,包含方向结论与该方向带回的 Evidence claim
    tool_payloads = [message for message in context if "research_direction" in message]
    assert len(tool_payloads) == 2
    assert all("直接事实" in message for message in tool_payloads)


def test_concurrent_delegates_allocate_unique_task_ids_before_absorb():
    """回归锁：序号必须分配即消耗，不能从已完成结果反推。

    旧实现下第二个并行 delegate 在第一个 absorb 之前读到同一 max+1，
    两个任务共享 task-0001，merge_task_results 按 task_id 去重静默吞掉
    一个方向（线上龙意象运行实锤）。交错点用 fake 研究代理入口让出一次
    事件循环来确定性复现。
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


def test_supervisor_adopts_higher_rank_stop_reason_when_dedup_thrash_hits_ceiling():
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
                    "stop_reason": "no_evidence",
                },
            }

    # complete_args=None 使 fake 每轮都重发同一批方向：首次执行、其余全部被问题级去重
    # 拦下(留下 NO_NEW_TASKS)。fake 不会主动收尾，最终是模型调用天花板掐断循环——
    # 声明式 rank 让更强的 MODEL_CALL_LIMIT_EXCEEDED 覆盖瞬时 NO_NEW_TASKS(rank 回归)。
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

    # 去重确实发生：只执行了一个方向，且未判充分、走部分报告。
    assert len(result["task_results"]) == 1
    assert result["supervisor"].is_sufficient is False
    # 终态归属真正的终止者：模型调用天花板压过去重瞬时信号。
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
                    "stop_reason": "no_evidence",
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

    assert result["supervisor"].is_sufficient is False
    assert result["supervisor_next"] == "render_final_report"
    assert "ResearchComplete 拒绝" in result["writer"].feedback


def test_supervisor_rejects_stale_complete_but_delivers_evidence_as_partial():
    class StaleCompleteLLM:
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
                            "args": {"research_topic": "新增证据方向"},
                            "id": "call_delegate",
                        }
                    ],
                )
            # 工作集已由 delegate 推进到 revision=1，但故意跳过综合稿修订。
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "ResearchComplete",
                        "args": {"synthesis_revision": 1, "reason": "错误使用旧版本"},
                        "id": "call_complete",
                    }
                ],
            )

    supervisor = ResearchSupervisor(
        StaleCompleteLLM(),
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

    assert result["working_set_revision"] == 1
    assert result["supervisor"].is_sufficient is False
    assert result["supervisor"].generation_mode == "partial"
    assert result["supervisor_next"] == "writer"
    assert result["writer_directive"].report_brief is not None
    assert "ResearchComplete 拒绝" in result["writer"].feedback


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
                            "args": {"research_topic": "新增证据方向"},
                            "id": "call_delegate",
                        }
                    ],
                )
            if '"status": "stale"' not in history:
                # delegate 的 absorb 已把工作集推进到 1；故意仍提交派发前看到的 0。
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
    assert "ResearchComplete 拒绝" in result["writer"].feedback


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
                            "args": {"research_topic": "证据将被释放的方向"},
                            "id": "call_delegate",
                        }
                    ],
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
    """agent loop 抛异常：异常文本走执行诊断与内部事件，不得伪装成"未闭合缺口"进报告。

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
        dedup_key=supervisor._task_deduplication_key,
        active_evidence_limit=30,
    )
    loop_state.stop_reason = StopReason.ROUND_BUDGET_EXHAUSTED

    result = supervisor._final_update(state, loop_state)

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
                            "args": {"research_topic": "补充方向"},
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
                    "args": {"research_topic": f"独立方向{self._index}"},
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
    """模型调用天花板触发时，终止原因如实记录，不再谎报轮次耗尽。"""

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
    """StopReason 是停止原因的唯一来源：词汇可回环、描述与兜底判定挂成员。"""
    from deepresearcher.schemas import StopReason

    for reason in StopReason:
        assert StopReason(reason.value) is reason  # 每个成员可由字符串值回环
        assert reason.description  # 无成员会拿空描述
        assert isinstance(reason, str)  # terminal_reason/JSON 等 str 消费点兼容

    # 兜底进入部分报告的集合语义钉死（原 _final_update 手工清单的行为锁）：
    assert {r for r in StopReason if r.allows_partial_report} == {
        StopReason.ROUND_BUDGET_EXHAUSTED,
        StopReason.GLOBAL_ROUND_BUDGET_EXHAUSTED,
        StopReason.MODEL_CALL_LIMIT_EXCEEDED,
        StopReason.NO_NEW_TASKS,
        StopReason.SUBMITTED_WITH_GAPS,
    }
    # 描述文案保留原逐字内容（回归锁）：
    assert StopReason.ROUND_BUDGET_EXHAUSTED.description == "研究轮次预算已耗尽。"
    assert StopReason.SUFFICIENT.description == "Supervisor 未确认现有材料足以形成完整研究报告。"


def test_stop_reason_rank_is_a_declared_total_order():
    """优先级是声明式的：每个成员有确定权威度，关键相邻关系被钉死。"""
    from deepresearcher.schemas import StopReason

    ranks = {reason: reason.rank for reason in StopReason}
    assert len(set(ranks.values())) == len(ranks)  # 无并列，全序确定
    # 修 bug 的那条：模型调用天花板必须压过瞬时去重信号。
    assert StopReason.MODEL_CALL_LIMIT_EXCEEDED.rank > StopReason.NO_NEW_TASKS.rank
    # 模型的显式收尾决定是最高终态，压过基础设施失败。
    assert StopReason.SUFFICIENT.rank > StopReason.AGENT_FAILED.rank
    assert StopReason.SUBMITTED_WITH_GAPS.rank > StopReason.AGENT_FAILED.rank
    # 全局轮次预算强于单轮，且都低于天花板命中。
    assert StopReason.GLOBAL_ROUND_BUDGET_EXHAUSTED.rank > StopReason.ROUND_BUDGET_EXHAUSTED.rank
    assert StopReason.MODEL_CALL_LIMIT_EXCEEDED.rank > StopReason.GLOBAL_ROUND_BUDGET_EXHAUSTED.rank


def test_loop_state_set_stop_reason_adopts_by_rank():
    """低权威度不覆盖高权威度；None 直接采纳；天花板覆盖瞬时去重信号（回归）。"""
    from deepresearcher.state import ResearchState

    state: ResearchState = {}
    loop_state = SupervisorLoopState(
        state, current_round=4, dedup_key=lambda q: q, active_evidence_limit=30
    )
    # current_round 由构造参数单一来源注入，不再从持久化 section 自读。
    assert loop_state.current_round == 4

    assert loop_state.stop_reason is None
    loop_state.set_stop_reason(StopReason.NO_NEW_TASKS)
    assert loop_state.stop_reason == StopReason.NO_NEW_TASKS

    # bug 场景：撞去重留下 NO_NEW_TASKS 后命中天花板 → 应升级为 MODEL_CALL_LIMIT。
    loop_state.set_stop_reason(StopReason.MODEL_CALL_LIMIT_EXCEEDED)
    assert loop_state.stop_reason == StopReason.MODEL_CALL_LIMIT_EXCEEDED

    # 更弱的瞬时信号不得回退覆盖已采纳的更强终止原因。
    loop_state.set_stop_reason(StopReason.NO_NEW_TASKS)
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
                "stop_reason": "no_evidence",
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
        dedup_key=supervisor._task_deduplication_key,
        active_evidence_limit=10,
    )
    context = SupervisorLoopContext(
        scope=AgentExecutionScope(run_id=run_id, agent_name="Supervisor"),
        deps=supervisor._deps,
        loop_state=loop_state,
    )
    return supervisor, context


def _delegate(context, topic: str) -> dict:
    """测试转发:与生产 tools.py 同一入口,参数全部来自 context 名词字段。"""
    return asyncio.run(
        services.delegate_research(
            context.deps, context.loop_state, context.scope, context.bookkeeping_lock, topic
        )
    )


def test_delegate_research_blocks_when_round_budget_exhausted():
    """轮次超过 max_research_rounds → 返回 blocked 且绝不派发子 Agent。"""
    agent = _RecordingDelegateAgent()
    supervisor, context = _delegate_context("run-x", current_round=5, agent=agent, max_rounds=3)
    result = _delegate(context, "一个方向")
    assert result["status"] == "blocked"
    assert result["reason"] == "round_budget_exhausted"
    assert agent.task_run_ids == []


def test_delegate_research_dedups_topics_and_scopes_run_id():
    """同 topic 第二次 skipped；任务 run_id 取自各自 context.scope.run_id，无跨 run 泄漏。"""
    agent_a = _RecordingDelegateAgent()
    sup_a, ctx_a = _delegate_context("run-a", current_round=1, agent=agent_a)
    first = _delegate(ctx_a, "重复主题")
    second = _delegate(ctx_a, "重复主题")
    assert first["status"] == "completed"
    assert second["status"] == "skipped" and second["reason"] == "duplicate_or_budget"
    assert agent_a.task_run_ids == ["run-a"]

    agent_b = _RecordingDelegateAgent()
    sup_b, ctx_b = _delegate_context("run-b", current_round=1, agent=agent_b)
    _delegate(ctx_b, "重复主题")
    assert agent_b.task_run_ids == ["run-b"]
