import asyncio
import hashlib

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from deepsearch_agent.agents.supervisor import ResearchSupervisor
from deepsearch_agent.config import AgentConfig
from deepsearch_agent.evidence.models import Evidence
from deepsearch_agent.schemas import (
    ResearchAgentResult,
    ResearchDirectionDecision,
    ResearchDirectionResult,
)
from fakes import evidence, researcher_agent


class SupervisorLLM:
    """工具调用协议的 Supervisor fake:按消息历史无状态决策,并发安全。

    - 历史中没有研究工具结果(以 ToolMessage 载荷标记 "research_direction")时:派发 ResearchDelegate;
    - 已有研究工具结果时:按 complete 参数决定输出 ResearchComplete(或继续派发)。

    bind_tools 返回的模型直接返回带 tool_calls 的 AIMessage。
    """

    def __init__(self, *, delegate_topics, complete_args=None):
        self.delegate_topics = delegate_topics
        self.complete_args = complete_args

    def bind_tools(self, _tools, tool_choice="any"):
        return self

    async def ainvoke(self, messages):
        history = "\n".join(str(message.content) for message in messages)
        if self.complete_args is not None and (
            "research_direction" in history or not self.delegate_topics
        ):
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
    async def run(self, task, *, claim_url, on_url_already_attempted=None):
        item = evidence(task["id"], task_id=task["id"])
        return ResearchAgentResult(
            evidences=[item],
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
    "report_brief": {
        "answer_goal": "回答测试问题",
        "covered_topics": [
            {"topic": "核心结论", "role": "主线", "reason": "直接回答问题", "required": True},
        ],
        "required_points": ["给出有来源的结论"],
        "caveats": [],
    },
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
    assert result["research"].is_sufficient is True
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
    from deepsearch_agent.observability.events import JsonlSink
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


def test_supervisor_url_deduplication_is_scoped_to_each_research_state():
    class UrlRecordingAgent:
        def __init__(self):
            self.claim_results: list[bool] = []

        async def run(self, task, *, claim_url, on_url_already_attempted=None):
            url = "https://example.com/shared#fragment"
            claimed = await claim_url(url)
            self.claim_results.append(claimed)
            if not claimed and on_url_already_attempted:
                on_url_already_attempted(url)
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

    agent = UrlRecordingAgent()
    supervisor = ResearchSupervisor(
        SupervisorLLM(
            delegate_topics=["验证共享来源"],
            complete_args=None,
        ),
        AgentConfig(max_research_rounds=1, max_parallel_workers=2),
        research_agent=agent,
    )

    async def run_two_sessions():
        return await asyncio.gather(
            supervisor.run({"query": "问题 A", "clarified_query": "问题 A", "evidences": []}),
            supervisor.run({"query": "问题 B", "clarified_query": "问题 B", "evidences": []}),
        )

    first, second = asyncio.run(run_two_sessions())

    assert agent.claim_results == [True, True]
    assert first["attempted_source_urls"] == ["https://example.com/shared"]
    assert second["attempted_source_urls"] == ["https://example.com/shared"]


def test_supervisor_stops_with_no_new_tasks_when_all_directions_are_deduplicated():
    class EmptyAgent:
        async def run(self, task, *, claim_url, on_url_already_attempted=None):
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

    # complete_args=None 使 fake 在每次调用都派发同一批方向;
    # 第二轮全部被问题级去重拦下 → no_new_tasks 停止,而不是空转烧轮次。
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

    assert len(result["task_results"]) == 1
    assert result["research"].is_sufficient is False
    assert "没有可去重的新研究任务" in result["writer"].feedback


def test_supervisor_rejects_completion_without_evidence():
    class EmptyAgent:
        async def run(self, task, *, claim_url, on_url_already_attempted=None):
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

    assert result["research"].is_sufficient is False
    assert result["supervisor_next"] == "render_final_report"
    assert "矛盾" in result["writer"].feedback


def test_supervisor_allows_partial_report_after_research_budget_exhaustion():
    supervisor = ResearchSupervisor(
        SupervisorLLM(delegate_topics=["一个局部方向"], complete_args=None),
        AgentConfig(
            max_research_rounds=1,
            partial_report_min_evidences=1,
            partial_report_min_sources=1,
        ),
            research_agent=researcher_agent(
                AgentConfig(research_agent_max_evidences_per_direction=1, research_agent_max_turns=3),
                [
                    ResearchDirectionDecision(action="search", reason="检索局部事实", queries=["局部事实"]),
                    ResearchDirectionDecision(
                        action="read",
                        reason="读取局部来源",
                        candidate_ids=[
                            "c-"
                            + hashlib.sha1("https://example.com/局部事实/a".encode()).hexdigest()[:10]
                        ],
                    ),
                    ResearchDirectionDecision(action="complete", reason="方向材料已收集"),
                ],
            ),
    )

    result = asyncio.run(
        supervisor.run({"query": "研究问题", "clarified_query": "研究问题", "evidences": []})
    )

    assert result["research"].is_sufficient is False
    assert result["research"].status == "incomplete"
    assert result["research"].generation_mode == "partial"
    assert result["supervisor_next"] == "writer"
    assert result["run"].phase == "writing"
    assert result["report_brief"] is not None


def test_supervisor_review_rejection_can_continue_research_via_tool_loop():
    class ReviewStateAwareLLM(SupervisorLLM):
        """看到审阅回流消息时补派一个方向,随后给出完整决策。"""

        def __init__(self):
            super().__init__(delegate_topics=["补充方向"], complete_args=_COMPLETE_ARGS)
            self._extra_delegated = False

        async def ainvoke(self, messages):
            history = "\n".join(str(message.content) for message in messages)
            if "【审阅回流】" in history and not self._extra_delegated:
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

    agent = researcher_agent(
        AgentConfig(research_agent_max_evidences_per_direction=1, research_agent_max_turns=3),
        [
            ResearchDirectionDecision(action="search", reason="检索", queries=["补充方向"]),
            ResearchDirectionDecision(
                action="read",
                reason="读取补充来源",
                candidate_ids=[
                    "c-"
                    + hashlib.sha1("https://example.com/补充方向/a".encode()).hexdigest()[:10]
                ],
            ),
            ResearchDirectionDecision(action="complete", reason="补充完成"),
        ],
    )
    supervisor = ResearchSupervisor(
        ReviewStateAwareLLM(),
        AgentConfig(max_research_rounds=1, max_post_review_recovery_cycles=1),
        research_agent=agent,
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

    assert result["research"].is_sufficient is True
    assert result["supervisor_next"] == "writer"
    assert len(result["task_results"]) == 1
    assert result["task_results"][0].question == "补充方向"


def test_supervisor_review_rejection_can_rewrite_without_extra_research():
    class RewriteAwareLLM(SupervisorLLM):
        def __init__(self):
            super().__init__(delegate_topics=[], complete_args=None)

        async def ainvoke(self, messages):
            history = "\n".join(str(message.content) for message in messages)
            if "【审阅回流】" in history:
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
                "review": {
                    "status": "rejected",
                    "attempts": 1,
                    "feedback": "措辞需要收窄。",
                    "gaps": [],
                },
            }
        )
    )

    assert result["research"].is_sufficient is True
    assert result["supervisor_next"] == "writer"
    assert result["task_results"] == []
    assert "审阅回流" in "\n".join(
        str(message.content) for message in result["supervisor_messages"]
    )


class _OneEvidenceAgent:
    """每个方向返回一条唯一 Evidence 的最小 ResearchAgent。"""

    def __init__(self):
        self.run_count = 0

    async def run(self, task, *, claim_url, on_url_already_attempted=None):
        self.run_count += 1
        item = evidence(f"t{self.run_count}", task_id=task["id"])
        return {
            "evidences": [item],
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
        if "limit exceeded" in tool_output or "round_budget_exhausted" in tool_output:
            return AIMessage(
                content="",
                tool_calls=[
                    {"name": "ResearchComplete", "args": _COMPLETE_ARGS, "id": "call_done"}
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
                "research": {"status": "incomplete", "current_round": 1},
            }
        )
    )

    assert agent.run_count == 0
    assert result["task_results"] == []
    assert "round_budget_exhausted" in "\n".join(
        str(message.content) for message in result["supervisor_messages"]
    )
    assert result["research"].is_sufficient is True
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
            partial_report_min_evidences=1,
            partial_report_min_sources=1,
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
    from deepsearch_agent.schemas import StopReason

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
    }
    # 描述文案保留原逐字内容（回归锁）：
    assert StopReason.ROUND_BUDGET_EXHAUSTED.description == "研究轮次预算已耗尽。"
    assert StopReason.SUFFICIENT.description == "Supervisor 未确认现有材料足以形成完整研究报告。"
