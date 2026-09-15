import asyncio
import hashlib

import pytest
from langchain_core.messages import AIMessage

from deepresearcher.agents.researcher import ResearchAgent
from deepresearcher.agents.researcher.state import DirectionRunState
from deepresearcher.config import AgentConfig
from deepresearcher.evidence.extractor import ExtractionResult
from deepresearcher.evidence.models import Evidence
from deepresearcher.llm import LLMConfigurationError
from deepresearcher.schemas import (
    ResearchDirectionDecision,
    ResearchDirectionResult,
)
from deepresearcher.tools import LocalDocumentStore, SearchTool, SourceReaderTool
from deepresearcher.tools.errors import SourceUnavailableError
from fakes import (
    TASK,
    DirectionLLM,
    FakeReader,
    FakeSearchClient,
    researcher_agent,
)


def test_direction_result_ignores_legacy_answered_points_on_checkpoint_restore() -> None:
    restored = ResearchDirectionResult.model_validate(
        {
            "task_id": "legacy-task",
            "round": 1,
            "question": "历史问题",
            "research_direction": "历史方向",
            "execution_status": "completed",
            "coverage_status": "partial",
            "evidence_count": 1,
            "source_count": 1,
            "answered_points": ["旧版字段"],
            "conclusion": "历史结论",
            "stop_reason": "complete",
        }
    )

    assert restored.conclusion == "历史结论"
    assert "answered_points" not in restored.model_dump()


def test_research_agent_autonomously_decides_queries_then_collects_direction_evidence():
    class CapturingReader(FakeReader):
        def __init__(self):
            super().__init__()
            self.tasks = []

        async def arun(self, task, candidate):
            self.tasks.append(task)
            return await super().arun(task, candidate)

    config = AgentConfig(
        research_agent_max_evidences_per_direction=2,
        research_agent_max_turns=4,
        research_agent_max_queries=4,
        research_agent_read_concurrency=2,
    )
    reader = CapturingReader()
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
    assert reader.tasks[0]["research_direction"] == TASK["question"]
    assert reader.tasks[0]["subquestions"] == ["稳定术语 定义", "稳定术语 直接证据"]


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
                    "c-" + hashlib.sha1("https://example.com/方向定义/a".encode()).hexdigest()[:10]
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
            ResearchDirectionDecision(
                action="release", reason="释放当前材料", evidence_ids=["r1-1-src-unknown-ev-1"]
            ),
            ResearchDirectionDecision(action="complete", reason="停止"),
        ],
    )
    # 该测试只验证工具协议和观察回流；ID 不匹配时 ReleaseEvidence 应安全返回 unknown。
    result = asyncio.run(agent.run(TASK, claim_url=lambda _url: _true()))
    assert result.task_result.execution_status == "completed"
    assert any(
        "工作集" in str(message.content)
        for snapshot in agent.llm.seen_messages
        for message in snapshot
    )


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
                conclusion="不应被接收的结论。",
            )
        ],
    )

    result = asyncio.run(agent.run(TASK, claim_url=lambda _url: _true()))

    task_result = result.task_result
    assert task_result.execution_status == "completed"
    assert task_result.stop_reason == "blocked_without_evidence"
    assert task_result.stop_detail == "我认为可以结束。"
    assert task_result.conclusion == ""
    assert task_result.remaining_gaps


def test_researcher_builds_minimum_result_when_finalization_never_submits():
    run_state = DirectionRunState(active_evidence_limit=2)
    run_state.add_evidences(
        [
            Evidence(
                evidence_id="e1",
                subtask_id="t1",
                research_direction="方向",
                claim="已验证的有限事实",
                quote="原文片段",
                source_url="https://example.com/a",
                support="direct",
            )
        ]
    )

    ResearchAgent._apply_minimum_result(run_state)

    assert run_state.stop_reason == "fallback_complete"
    assert run_state.conclusion.startswith("本方向未完成模型综合")
    assert run_state.remaining_gaps


def test_researcher_builds_blocked_minimum_result_without_evidence():
    run_state = DirectionRunState()

    ResearchAgent._apply_minimum_result(run_state)

    assert run_state.stop_reason == "blocked_without_evidence"
    assert run_state.conclusion == ""
    assert "未获得可用 Evidence。" in run_state.remaining_gaps


def test_researcher_exhaustion_preserves_collected_evidence_as_minimum_result():
    agent = researcher_agent(
        AgentConfig(
            research_agent_max_turns=1,
            finalization_attempts=1,
            research_agent_max_evidences_per_direction=2,
        ),
        [
            ResearchDirectionDecision(action="search", reason="先找来源", queries=["方向"]),
            ResearchDirectionDecision(
                action="read",
                reason="读取来源",
                candidate_ids=[
                    "c-" + hashlib.sha1("https://example.com/方向/a".encode()).hexdigest()[:10]
                ],
            ),
        ],
    )

    result = asyncio.run(agent.run(TASK, claim_url=lambda _url: _true()))

    assert result.evidences
    assert result.task_result.execution_status == "completed"
    assert result.task_result.stop_reason == "fallback_complete"
    assert result.task_result.conclusion.startswith("本方向未完成模型综合")


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


def test_source_reader_inlines_short_document_and_registers_it(tmp_path):
    class Fetcher:
        async def afetch(self, _url, **_kwargs):
            return {
                "title": "短文",
                "text": "第一行\n第二行",
                "final_url": "https://example.com/final",
                "retrieval_method": "origin_fetch",
            }

    store = LocalDocumentStore(tmp_path)
    tool = SourceReaderTool(
        Fetcher(),
        llm=object(),
        document_store=store,
        document_inline_max_tokens=1_000,
    )
    result = asyncio.run(
        tool.arun(TASK, {"title": "x", "url": "https://example.com", "score": 0.9})
    )

    assert result.status == "completed"
    assert result.evidences == []
    assert result.documents[0].inline is True
    assert result.documents[0].content == "L1: 第一行\nL2: 第二行"
    assert asyncio.run(store.get(result.documents[0].document_id)).source_url.endswith("/final")


def test_source_reader_keeps_long_document_out_of_tool_result(tmp_path):
    class Fetcher:
        async def afetch(self, _url, **_kwargs):
            return {"title": "长文", "text": "很长的正文" * 100, "final_url": _url}

    tool = SourceReaderTool(
        Fetcher(),
        llm=object(),
        document_store=LocalDocumentStore(tmp_path),
        document_inline_max_tokens=1,
    )
    result = asyncio.run(
        tool.arun(TASK, {"title": "x", "url": "https://example.com", "score": 0.9})
    )

    assert result.status == "completed"
    assert result.documents[0].inline is False
    assert result.documents[0].content == ""
    assert result.documents[0].line_count == 1


def test_researcher_reads_registered_document_and_adds_verified_evidence(tmp_path):
    store = LocalDocumentStore(tmp_path)
    document = asyncio.run(
        store.put(
            text="引言\n实验表明端到端延迟为 20ms。\n结论",
            title="实验报告",
            source_url="https://example.com/report",
            support_ceiling="partial",
            token_count=20,
        )
    )
    agent = ResearchAgent(
        DirectionLLM([]),
        AgentConfig(evidence_max_per_source=4),
        search_tool=SearchTool(FakeSearchClient()),
        reader_tool=FakeReader(),
        document_store=store,
    )
    run_state = DirectionRunState(active_evidence_limit=4, evidence_archive_limit=8)
    run_state.documents[document.document_id] = document

    grep = asyncio.run(
        agent._grep_document(
            document.document_id,
            ["延迟"],
            1,
            "定位数据",
            run_state=run_state,
        )
    )
    read = asyncio.run(
        agent._read_document(
            document.document_id,
            [(2, 3)],
            "读取数据",
            run_state=run_state,
        )
    )
    added = asyncio.run(
        agent._add_evidence(
            TASK,
            [
                {
                    "document_id": document.document_id,
                    "claim": "实验端到端延迟为 20ms。",
                    "quote": "实验表明端到端延迟为 20ms。",
                    "support": "direct",
                    "confidence": 0.9,
                },
                {
                    "document_id": document.document_id,
                    "claim": "不存在的事实",
                    "quote": "正文里没有这句话",
                    "support": "direct",
                    "confidence": 0.9,
                },
            ],
            "批量提交",
            run_state=run_state,
            event_context={},
        )
    )

    assert grep["match_count"] == 1
    assert "L2:" in read["ranges"][0]["content"]
    assert len(added["accepted"]) == 1
    assert len(added["rejected"]) == 1
    assert run_state.evidences[0].support == "partial"  # 不得突破来源支撑上限
    assert run_state.evidences[0].locator.start_line == 2
    assert run_state.evidences[0].locator.end_line == 2


def test_add_evidence_validates_before_ranking_by_confidence(tmp_path):
    store = LocalDocumentStore(tmp_path)
    document = asyncio.run(
        store.put(
            text="低优先级事实\n高优先级事实",
            title="排序测试",
            source_url="https://example.com/ranking",
        )
    )
    agent = ResearchAgent(
        DirectionLLM([]),
        AgentConfig(evidence_add_batch_size=1, evidence_max_per_source=2),
        search_tool=SearchTool(FakeSearchClient()),
        reader_tool=FakeReader(),
        document_store=store,
    )
    run_state = DirectionRunState(active_evidence_limit=2, evidence_archive_limit=2)
    run_state.documents[document.document_id] = document
    run_state.observed_document_ranges[document.document_id] = [(1, 2)]

    result = asyncio.run(
        agent._add_evidence(
            TASK,
            [
                {
                    "document_id": document.document_id,
                    "claim": "无效但最高分",
                    "quote": "并不存在",
                    "confidence": 1.0,
                },
                {
                    "document_id": document.document_id,
                    "claim": "较低优先级",
                    "quote": "低优先级事实",
                    "confidence": 0.2,
                },
                {
                    "document_id": document.document_id,
                    "claim": "较高优先级",
                    "quote": "高优先级事实",
                    "confidence": 0.9,
                },
            ],
            "验证后排序",
            run_state=run_state,
            event_context={},
        )
    )

    assert len(result["accepted"]) == 1
    assert run_state.evidences[0].claim == "较高优先级"
    assert result["truncated_submission_count"] == 1
    assert any(item["reason"].endswith("不存在于候选原文") for item in result["rejected"])


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
            assert document["published_at"] == "2026-07-04"  # 搜索元信息随退化文档下传
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
                "published_at": "2026-07-04",
            },
        )
    )

    assert result.status == "completed"
    assert result.evidences[0].retrieval_method == "tavily_raw_content"
    assert result.evidences[0].support == "direct"


def test_source_reader_attaches_search_published_at_to_fetched_document():
    class OkFetcher:
        async def afetch(self, _url, **_kwargs):
            return {"text": "正文", "final_url": "https://example.com"}

    class CapturingExtractor:
        async def aextract_result(self, task, document, result):
            assert document["published_at"] == "2026-01-02"  # 成功抓取路径同样透传
            return ExtractionResult(
                evidences=[
                    Evidence(
                        evidence_id="r1-1-ev-1",
                        subtask_id=task["id"],
                        research_direction=task["question"],
                        claim="正文支持的事实",
                        quote=document["text"],
                        source_url=result["url"],
                        retrieval_method="origin_fetch",
                        support="direct",
                    )
                ],
                strategy="full_document",
                chunk_count=1,
                candidate_chars=len(document["text"]),
            )

    tool = SourceReaderTool(OkFetcher(), llm=object())
    tool.extractor = CapturingExtractor()
    result = asyncio.run(
        tool.arun(
            TASK,
            {"title": "t", "url": "https://example.com", "published_at": "2026-01-02"},
        )
    )
    assert result.status == "completed"  # 空 evidence 会被生产语义判 skipped，故须回一条
    assert result.evidences[0].evidence_id == "r1-1-ev-1"


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


def test_direction_evidence_pool_releases_slots_without_deleting_archive():
    pool = DirectionRunState(active_evidence_limit=2, evidence_archive_limit=4)
    first = [
        Evidence(
            evidence_id=f"e{index}",
            subtask_id="task-1",
            research_direction="方向",
            claim=f"事实 {index}",
            quote=f"原文 {index}",
            source_url=f"https://example.com/{index}",
        )
        for index in range(1, 5)
    ]

    pool.add_evidences(first[:2])
    assert pool.active_evidence_ids == {"e1", "e2"}

    assert pool.release_evidence(["e1"]) == ["e1"]
    pool.add_evidences(first[2:])

    assert [item.evidence_id for item in pool.evidences] == ["e1", "e2", "e3", "e4"]
    assert pool.active_evidence_ids == {"e2", "e3"}
    assert pool.restore_evidence(["e1"]) == []  # 活跃槽位已满

    pool.release_evidence(["e2"])
    assert pool.restore_evidence(["e1"]) == ["e1"]
    assert pool.active_evidence_ids == {"e1", "e3"}
