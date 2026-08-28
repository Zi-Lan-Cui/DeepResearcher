import asyncio
import hashlib

import pytest
from langchain_core.messages import AIMessage

from deepsearch_agent.agents.researcher import ResearchAgent
from deepsearch_agent.config import AgentConfig
from deepsearch_agent.evidence.extractor import ExtractionResult
from deepsearch_agent.evidence.models import Evidence
from deepsearch_agent.llm import LLMConfigurationError
from deepsearch_agent.schemas import (
    ResearchDirectionDecision,
)
from deepsearch_agent.tools import SearchTool, SourceReaderTool
from deepsearch_agent.tools.errors import SourceUnavailableError
from fakes import (
    TASK,
    DirectionLLM,
    FakeReader,
    FakeSearchClient,
    researcher_agent,
)


def test_research_agent_autonomously_decides_queries_then_collects_direction_evidence():
    config = AgentConfig(
        research_agent_max_evidences_per_direction=2,
        research_agent_max_turns=4,
        research_agent_max_queries=4,
        research_agent_read_concurrency=2,
    )
    reader = FakeReader()
    agent = researcher_agent(
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
    agent = researcher_agent(
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
    agent = researcher_agent(
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
    agent = researcher_agent(
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
    agent = researcher_agent(
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
    agent = researcher_agent(
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


async def _true() -> bool:
    return True
