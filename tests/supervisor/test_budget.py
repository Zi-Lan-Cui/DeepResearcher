"""轮次与调用预算的执行边界，以及终止原因的排序与对外文案。"""

import asyncio
import re

from langchain_core.messages import AIMessage, ToolMessage

from deepresearcher.agents.supervisor import Supervisor, services
from deepresearcher.agents.supervisor.state import (
    SupervisorRoundState,
)
from deepresearcher.config import AgentConfig
from deepresearcher.schemas import (
    StopReason,
)
from fakes import evidence
from supervisor_kit import (
    FixedResearcher,
    SupervisorLLM,
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

    # complete_args=None 使 fake 每轮重发同一批方向：产品不做程序化去重，重复
    # 方向每次都真实执行。fake 不会主动收尾，最终由模型调用次数上限终止循环。
    # 声明式 rank 让更强的 MODEL_CALL_LIMIT_EXCEEDED 压过轮次预算(终态归属回归)。
    supervisor = Supervisor(
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


def test_freeze_without_draft_loops_to_model_call_ceiling():
    """无草稿冻结被拒且留环恢复；模型反复不成时，终态归属模型调用上限。"""

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
                    tool_calls=[{"name": "WaitTask", "args": {}, "id": "call_wait"}],
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

    supervisor = Supervisor(
        NoDraftFreezeLLM(),
        AgentConfig(max_research_rounds=1),
        research_agent=FixedResearcher(),
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
    # 协议拒绝不得污染收尾诊断：它只解释"为什么停"，不描述研究内容。
    assert "ResearchComplete" not in result["writer"].feedback


def test_supervisor_allows_partial_report_after_research_budget_exhaustion():
    supervisor = Supervisor(
        SupervisorLLM(delegate_topics=["一个局部方向"], ready=True),
        AgentConfig(max_research_rounds=1),
        research_agent=FixedResearcher(),
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
    supervisor = Supervisor(
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
    round_state = SupervisorRoundState(
        state,
        current_round=0,
        active_evidence_limit=30,
    )
    round_state.stop_reason = StopReason.ENDED_WITHOUT_COMPLETION

    result = services.compose_final_update(state, round_state, supervisor.config)

    assert result.run.phase == "writing"
    assert result.supervisor.generation_mode == "partial"
    assert result.research_synthesis is not None
    assert result.research_synthesis.revision == 1
    assert result.writer_directive is not None
    assert result.writer_directive.report_brief.chapters[0].evidence_ids == [item.evidence_id]
    assert result.writer_directive.evidence_ids == [item.evidence_id]


class _OneEvidenceAgent:
    """每个方向返回一条唯一 Evidence 的最小 Researcher。"""

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
    supervisor = Supervisor(
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
    supervisor = Supervisor(
        _DelegateUntilBlockedLLM(),
        AgentConfig(
            max_research_rounds=3,
            max_tasks_per_round=2,
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

    supervisor = Supervisor(
        ReadLoopLLM(),
        AgentConfig(max_research_rounds=3, max_tasks_per_round=1),
        research_agent=_OneEvidenceAgent(),
    )
    result = asyncio.run(
        supervisor.run({"query": "研究问题", "clarified_query": "研究问题", "evidences": []})
    )

    assert result["run"].terminal_reason == "supervisor_model_call_limit_exceeded"
    assert "模型调用预算已耗尽" in result["writer"].feedback


def test_stop_reason_vocabulary_single_source():
    """StopReason 是停止原因的唯一来源：值可回环构造，描述与默认值判定都在成员上。"""
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


def test_round_state_set_stop_reason_adopts_by_rank():
    """低权威度不覆盖高权威度；None 直接采纳；更强信号后到则升级覆盖。"""
    from deepresearcher.state import RunState

    state: RunState = {}
    round_state = SupervisorRoundState(state, current_round=4, active_evidence_limit=30)
    # current_round 只由构造参数注入，持久化 section 不作为第二个来源。
    assert round_state.current_round == 4

    assert round_state.stop_reason is None
    round_state.set_stop_reason(StopReason.ENDED_WITHOUT_COMPLETION)
    assert round_state.stop_reason == StopReason.ENDED_WITHOUT_COMPLETION

    # 先记录轮次预算、随后命中调用上限 → 应升级为 MODEL_CALL_LIMIT。
    round_state.set_stop_reason(StopReason.MODEL_CALL_LIMIT_EXCEEDED)
    assert round_state.stop_reason == StopReason.MODEL_CALL_LIMIT_EXCEEDED

    # 更弱的信号不得回退覆盖已采纳的更强终止原因。
    round_state.set_stop_reason(StopReason.ENDED_WITHOUT_COMPLETION)
    assert round_state.stop_reason == StopReason.MODEL_CALL_LIMIT_EXCEEDED

    # 模型的显式收尾决定压过一切基础设施信号。
    round_state.set_stop_reason(StopReason.COMPLETED)
    assert round_state.stop_reason == StopReason.COMPLETED


def test_render_outcome_vocabulary_and_user_text_lookup():
    """terminal_reason 三域词汇的出口规则：引擎两域查表成中文，lifecycle 原样。"""
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
