import asyncio
import hashlib

import pytest
from langchain_core.messages import AIMessage

from deepsearch_agent.agents.researcher import ResearchAgent
from deepsearch_agent.agents.supervisor import ResearchSupervisor
from deepsearch_agent.config import AgentConfig
from deepsearch_agent.evidence.extractor import ExtractionResult
from deepsearch_agent.evidence.models import Evidence
from deepsearch_agent.llm import LLMConfigurationError
from deepsearch_agent.schemas import (
    ForgetEvidence,
    ReadWorkingSet,
    ResearchAgentResult,
    ResearchDirectionComplete,
    ResearchDirectionDecision,
    ResearchDirectionResult,
    SearchSources,
)
from deepsearch_agent.state import SubTask
from deepsearch_agent.tools import SearchTool, SourceReaderTool
from deepsearch_agent.tools.errors import SourceUnavailableError

TASK: SubTask = {
    "id": "r1-1",
    "question": "验证一个具体研究方向",
    "type": "search",
    "status": "pending",
    "assigned_agent": "research_agent",
}


def evidence(evidence_id: str, *, task_id: str) -> Evidence:
    """构造 Supervisor 测试使用的最小有效 Evidence。"""
    return Evidence(
        evidence_id=f"{evidence_id}-ev",
        subtask_id=task_id,
        research_direction="测试方向",
        claim=f"{evidence_id} 的直接事实",
        quote="可验证原文。",
        source_url=f"https://example.com/{evidence_id}",
    )

class DirectionLLM:
    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.seen_messages: list[list] = []

    def bind_tools(self, _tools, **_kwargs):
        return self

    async def ainvoke(self, _messages):
        self.seen_messages.append(list(_messages))
        decision = next(self.decisions)
        if decision.action == "search":
            args = SearchSources(
                reason=decision.reason,
                queries=decision.queries,
            ).model_dump()
            name = "SearchSources"
        elif decision.action == "read":
            args = {
                "reason": decision.reason,
                "candidate_ids": decision.candidate_ids,
            }
            name = "ReadSources"
        elif decision.action == "inspect":
            args = ReadWorkingSet(reason=decision.reason).model_dump()
            name = "ReadWorkingSet"
        elif decision.action == "forget":
            args = ForgetEvidence(
                evidence_ids=decision.evidence_ids,
                reason=decision.reason,
            ).model_dump()
            name = "ForgetEvidence"
        else:
            args = ResearchDirectionComplete(
                reason=decision.reason,
                answered_points=decision.answered_points,
                conclusion=decision.conclusion,
                remaining_gaps=decision.remaining_gaps,
            ).model_dump()
            name = "ResearchDirectionComplete"
        return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "call"}])

    async def ainvoke_structured(self, schema, messages, **_kwargs):
        assert schema is ResearchDirectionDecision
        self.seen_messages.append(list(messages))
        return next(self.decisions)


class FakeSearchClient:
    async def asearch(self, query):
        return [
            {"title": query, "url": f"https://example.com/{query}/a", "snippet": "", "score": 0.9},
            {"title": query, "url": f"https://example.com/{query}/b", "snippet": "", "score": 0.8},
        ]


class FakeReader:
    def __init__(self):
        self.active = 0
        self.max_active = 0

    async def arun(self, _task, candidate):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.01)
            return {
                "task_id": _task["id"],
                "status": "completed",
                "source_url": candidate["url"],
                "evidences": [
                    Evidence(
                        evidence_id=f"ev-{len(candidate['url'])}",
                        subtask_id=_task["id"],
                        research_direction=_task["question"],
                        claim=f"{candidate['title']} 的直接事实",
                        quote="可验证原文",
                        source_url=candidate["url"],
                        support="direct",
                    )
                ],
            }
        finally:
            self.active -= 1


def _agent(config, decisions, reader=None):
    return ResearchAgent(
        DirectionLLM(decisions),
        config,
        search_tool=SearchTool(FakeSearchClient()),
        reader_tool=reader or FakeReader(),
    )


def test_research_agent_autonomously_decides_queries_then_collects_direction_evidence():
    config = AgentConfig(
        research_agent_max_evidences_per_direction=2,
        research_agent_max_turns=4,
        research_agent_max_queries=4,
        research_agent_read_concurrency=2,
    )
    reader = FakeReader()
    agent = _agent(
        config,
        [
            ResearchDirectionDecision(
                action="search",
                reason="尚无证据，先检索定义与直接事实。",
                queries=["稳定术语 定义", "稳定术语 直接证据"],
                remaining_gaps=["需要直接来源"],
            ),
            ResearchDirectionDecision(
                action="read",
                reason="读取直接来源。",
                candidate_ids=[
                    "c-"
                    + hashlib.sha1("https://example.com/稳定术语 定义/a".encode()).hexdigest()[:10]
                ],
            ),
            ResearchDirectionDecision(
                action="complete",
                reason="已获得直接来源。",
                answered_points=["获得方向所需的直接事实"],
                conclusion="当前方向可由已获取 Evidence 谨慎回答。",
            ),
        ],
        reader,
    )

    result = asyncio.run(agent.run(TASK, claim_url=lambda _url: _true()))

    task_result = result.task_result
    assert task_result.execution_status == "completed"
    assert task_result.queries == ["稳定术语 定义", "稳定术语 直接证据"]
    assert task_result.stop_reason == "complete"
    assert task_result.research_direction == TASK["question"]
    assert len(result.evidences) == 1
    assert reader.max_active == 1


def test_research_agent_records_context_observations_and_tool_results():
    agent = _agent(
            AgentConfig(
                research_agent_max_evidences_per_direction=2,
                research_agent_max_turns=3,
            ),
            [
                ResearchDirectionDecision(action="search", reason="先查直接事实", queries=["方向定义"]),
                ResearchDirectionDecision(
                    action="read",
                    reason="读取候选来源",
                    candidate_ids=[
                        "c-"
                        + hashlib.sha1("https://example.com/方向定义/a".encode()).hexdigest()[:10]
                    ],
                ),
                ResearchDirectionDecision(action="complete", reason="材料已足够"),
        ],
    )

    result = asyncio.run(agent.run(TASK, claim_url=lambda _url: _true()))
    assert result.evidences
    contents = [
        str(message.content) for snapshot in agent.llm.seen_messages for message in snapshot
    ]
    assert any("委派研究方向" in content for content in contents)
    assert any("系统研究观察" in content for content in contents)
    assert any("系统工具执行结果" in content for content in contents)
    assert any(
        isinstance(message, AIMessage)
        and any(call["name"] == "SearchSources" for call in message.tool_calls)
        for snapshot in agent.llm.seen_messages
        for message in snapshot
    )

def test_research_agent_can_inspect_and_forget_its_working_set():
    agent = _agent(
        AgentConfig(research_agent_max_evidences_per_direction=2, research_agent_max_turns=5),
        [
            ResearchDirectionDecision(action="search", reason="先找来源", queries=["方向"]),
            ResearchDirectionDecision(
                action="read",
                reason="读取来源",
                candidate_ids=[
                    "c-" + hashlib.sha1("https://example.com/方向/a".encode()).hexdigest()[:10]
                ],
            ),
            ResearchDirectionDecision(action="inspect", reason="确认工作集"),
            ResearchDirectionDecision(action="forget", reason="释放当前材料", evidence_ids=["r1-1-src-unknown-ev-1"]),
            ResearchDirectionDecision(action="complete", reason="停止"),
        ],
    )
    # 该测试只验证工具协议和观察回流；ID 不匹配时 ForgetEvidence 应安全返回 unknown。
    result = asyncio.run(agent.run(TASK, claim_url=lambda _url: _true()))
    assert result.task_result.execution_status == "completed"
    assert any("工作集" in str(message.content) for snapshot in agent.llm.seen_messages for message in snapshot)

def test_research_agent_can_stop_a_direction_without_unnecessary_search():
    agent = _agent(
        AgentConfig(),
        [
            ResearchDirectionDecision(
                action="complete",
                reason="没有可信的可检索路径。",
                remaining_gaps=["缺少可公开验证的来源"],
            )
        ],
    )

    result = asyncio.run(agent.run(TASK, claim_url=lambda _url: _true()))

    assert result.task_result.execution_status == "completed"
    assert result.task_result.stop_reason == "blocked_without_evidence"
    assert result.task_result.remaining_gaps == ["缺少可公开验证的来源"]


def test_research_agent_converts_completion_without_evidence_to_blocked_result():
    agent = _agent(
        AgentConfig(),
        [
            ResearchDirectionDecision(
                action="complete",
                reason="我认为可以结束。",
                answered_points=["不应被接收的回答点"],
                conclusion="不应被接收的结论。",
            )
        ],
    )

    result = asyncio.run(agent.run(TASK, claim_url=lambda _url: _true()))

    task_result = result.task_result
    assert task_result.execution_status == "completed"
    assert task_result.stop_reason == "blocked_without_evidence"
    assert task_result.stop_detail == "我认为可以结束。"
    assert task_result.answered_points == []
    assert task_result.conclusion == ""
    assert task_result.remaining_gaps


def test_research_direction_decision_rejects_conclusions_during_search():
    with pytest.raises(ValueError, match="只有 action=complete"):
        ResearchDirectionDecision(
            action="search",
            reason="还要继续检索。",
            queries=["新检索式"],
            conclusion="不应在搜索阶段给出结论。",
        )


def test_research_agent_replans_duplicate_queries_instead_of_mislabeling_budget_exhaustion():
    config = AgentConfig(
        research_agent_max_turns=3,
        research_agent_max_queries=4,
        research_agent_max_evidences_per_direction=4,
    )
    agent = _agent(
        config,
        [
            ResearchDirectionDecision(action="search", reason="先检索", queries=["稳定术语 定义"]),
            ResearchDirectionDecision(
                action="search", reason="误重复旧检索式", queries=["稳定术语 定义"]
            ),
            ResearchDirectionDecision(action="complete", reason="现有材料足以谨慎回答。"),
        ],
    )

    result = asyncio.run(agent.run(TASK, claim_url=lambda _url: _true()))

    assert result.task_result.stop_reason == "blocked_without_evidence"
    assert result.task_result.queries == ["稳定术语 定义"]
    assert "no_novel_queries" in result.task_result.failures[0]


def test_research_agent_requires_all_dependencies_at_construction():
    with pytest.raises(LLMConfigurationError):
        ResearchAgent(
            None,
            AgentConfig(),
            search_tool=SearchTool(FakeSearchClient()),
            reader_tool=FakeReader(),
        )
    with pytest.raises(ValueError, match="SearchTool"):
        ResearchAgent(DirectionLLM([]), AgentConfig(), search_tool=None, reader_tool=FakeReader())


def test_source_reader_requires_llm_at_construction():
    class Fetcher:
        async def afetch(self, _url, **_kwargs):
            return {"text": "正文"}

    with pytest.raises(LLMConfigurationError):
        SourceReaderTool(Fetcher(), llm=None)


def test_source_reader_keeps_access_challenge_as_nonfatal_source_outcome():
    class ChallengeFetcher:
        async def afetch(self, _url, **_kwargs):
            raise SourceUnavailableError("access_challenge", "来源返回验证码页")

    tool = SourceReaderTool(ChallengeFetcher(), llm=object())
    result = asyncio.run(
        tool.arun(TASK, {"title": "x", "url": "https://example.com", "score": 0.9})
    )
    assert result.status == "skipped"
    assert result.reason_code == "access_challenge"


def test_source_reader_uses_tavily_raw_content_when_page_fetch_is_unavailable():
    class BlockedFetcher:
        async def afetch(self, _url, **_kwargs):
            raise SourceUnavailableError("access_challenge", "来源返回验证码页")

    class CapturingExtractor:
        async def aextract_result(self, task, document, result):
            assert document["retrieval_method"] == "tavily_raw_content"
            assert document["support_ceiling"] == "direct"
            return ExtractionResult(
                evidences=[
                    Evidence(
                        evidence_id="r1-1-ev-1",
                        subtask_id=task["id"],
                        research_direction=task["question"],
                        claim="来源正文直接支持的事实",
                        quote=document["text"],
                        source_url=result["url"],
                        retrieval_method=document["retrieval_method"],
                        support=document["support_ceiling"],
                    )
                ],
                strategy="full_document",
                chunk_count=1,
                candidate_chars=len(document["text"]),
            )

    tool = SourceReaderTool(BlockedFetcher(), llm=object())
    tool.extractor = CapturingExtractor()
    result = asyncio.run(
        tool.arun(
            TASK,
            {
                "title": "来源",
                "url": "https://example.com",
                "raw_content": "提供商提取的来源正文。",
                "snippet": "短摘要",
                "content_provider": "tavily",
            },
        )
    )

    assert result.status == "completed"
    assert result.evidences[0].retrieval_method == "tavily_raw_content"
    assert result.evidences[0].support == "direct"


def test_source_reader_caps_search_summary_evidence_at_partial_support():
    class BlockedFetcher:
        async def afetch(self, _url, **_kwargs):
            raise SourceUnavailableError("access_challenge", "来源返回验证码页")

    class CapturingExtractor:
        async def aextract_result(self, task, document, result):
            assert document["retrieval_method"] == "search_summary"
            assert document["support_ceiling"] == "partial"
            return ExtractionResult(
                evidences=[
                    Evidence(
                        evidence_id="r1-1-ev-1",
                        subtask_id=task["id"],
                        research_direction=task["question"],
                        claim="搜索摘要的事实",
                        quote=document["text"],
                        source_url=result["url"],
                        retrieval_method=document["retrieval_method"],
                        support=document["support_ceiling"],
                    )
                ],
                strategy="full_document",
                chunk_count=1,
                candidate_chars=len(document["text"]),
            )

    tool = SourceReaderTool(BlockedFetcher(), llm=object())
    tool.extractor = CapturingExtractor()
    result = asyncio.run(
        tool.arun(
            TASK,
            {
                "title": "来源",
                "url": "https://example.com",
                "snippet": "搜索结果摘要。",
                "content_provider": "tavily",
            },
        )
    )

    assert result.status == "completed"
    assert result.evidences[0].retrieval_method == "search_summary"
    assert result.evidences[0].support == "partial"


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
            research_agent=_agent(
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

    agent = _agent(
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


async def _true() -> bool:
    return True
