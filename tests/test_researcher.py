import asyncio
import hashlib

import pytest
from langchain_core.messages import AIMessage

from deepresearcher.agents.researcher import ResearchAgent, services
from deepresearcher.agents.researcher.state import ResearcherLoopState
from deepresearcher.config import AgentConfig
from deepresearcher.evidence.models import Evidence
from deepresearcher.llm import LLMConfigurationError
from deepresearcher.schemas import (
    ResearchDirectionDecision,
    ResearchDirectionResult,
)
from deepresearcher.schemas.limits import SEARCH_RESULTS_PREVIEW_COUNT
from deepresearcher.tools import SearchTool, SourceReaderTool
from deepresearcher.tools.errors import SourceUnavailableError
from deepresearcher.tools.web.materials import MemoryResearchMaterialStore
from fakes import (
    TASK,
    DirectionLLM,
    FakeReader,
    FakeSearchService,
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


def test_research_agent_search_and_read_do_not_implicitly_create_evidence():
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

    result = asyncio.run(agent.run(TASK))

    task_result = result.task_result
    assert task_result.execution_status == "completed"
    assert task_result.queries == ["稳定术语 定义", "稳定术语 直接证据"]
    assert task_result.stop_reason == "blocked_without_evidence"
    assert task_result.research_direction == TASK["question"]
    assert result.evidences == []
    assert reader.max_active == 1
    assert reader.tasks[0] == TASK


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

    result = asyncio.run(agent.run(TASK))
    assert result.evidences == []
    assert result.task_result.stop_reason == "blocked_without_evidence"
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


def test_search_results_are_compact_and_can_be_paged_from_search_cache():
    class ManySearchService:
        def __init__(self):
            self.calls = 0

        async def asearch(self, query):
            self.calls += 1
            return [
                {
                    "title": f"{query}-{index}",
                    "url": f"https://example.com/{index}",
                    "snippet": f"摘要-{index}-" + "x" * 5_000,
                    "score": 1.0 - index / 100,
                }
                for index in range(12)
            ]

    client = ManySearchService()
    materials = MemoryResearchMaterialStore()
    agent = ResearchAgent(
        DirectionLLM([]),
        AgentConfig(),
        search_tool=SearchTool(client),
        reader_tool=FakeReader(),
        material_store=materials,
    )
    loop_state = ResearcherLoopState()
    first = asyncio.run(
        services.search_sources(agent._deps, TASK, loop_state, {}, ["分页测试"], "发现候选")
    )

    assert first["result_count"] == 12
    assert len(first["candidates"]) == SEARCH_RESULTS_PREVIEW_COUNT
    assert first["has_more"] is True
    # 预览与分页统一形状：首屏即带（截断）snippet，模型不必再翻 ListSearchResults(offset=0) 才能判断。
    assert all(item["snippet"] and len(item["snippet"]) <= 1_200 for item in first["candidates"])
    assert first["next_offset"] == SEARCH_RESULTS_PREVIEW_COUNT  # 预览消费 0..N-1 → 下一页从 N

    page = asyncio.run(
        services.list_search_results(
            agent._deps, TASK, loop_state, first["search_id"], 5, 3, "查看下一页"
        )
    )

    assert page["offset"] == 5
    assert page["returned_count"] == 3
    assert page["next_offset"] == 8
    # 同一候选在预览与分页里字段完全一致（无 include_snippet 分叉）。
    assert set(first["candidates"][0]) == set(page["candidates"][0])
    # 分页重建只从 SearchTool 的进程/持久缓存取数，不再请求 Provider。
    assert client.calls == 1
    stored = asyncio.run(
        materials.get_search_results(str(TASK.get("run_id") or TASK["id"]), first["search_id"])
    )
    assert len(stored.results) == 12


def test_source_reader_reuses_document_from_material_store():
    class Fetcher:
        def __init__(self):
            self.calls = 0

        def material_fetch_key(self, url):
            return f"fetch:{url}"

        async def afetch(self, url, **_kwargs):
            self.calls += 1
            return {"title": "cached", "text": "first\nsecond", "final_url": url}

    fetcher = Fetcher()
    materials = MemoryResearchMaterialStore()
    tool = SourceReaderTool(
        fetcher,
        material_store=materials,
        document_inline_max_tokens=1_000,
    )
    first = asyncio.run(tool.arun(TASK, {"url": "https://example.com"}))
    second = asyncio.run(tool.arun(TASK, {"url": "https://example.com"}))

    assert first.status == second.status == "completed"
    assert first.documents[0].document_id == second.documents[0].document_id
    assert second.documents[0].content == "L1: first\nL2: second"
    assert fetcher.calls == 1


def test_list_search_results_rejects_another_direction_handle():
    agent = researcher_agent(AgentConfig(), [])
    result = asyncio.run(
        services.list_search_results(
            agent._deps, TASK, ResearcherLoopState(), "search-not-owned", 0, 5, "越权查看"
        )
    )

    assert result == {"status": "rejected", "reason": "unknown_search_id"}


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
    result = asyncio.run(agent.run(TASK))
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

    result = asyncio.run(agent.run(TASK))

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

    result = asyncio.run(agent.run(TASK))

    task_result = result.task_result
    assert task_result.execution_status == "completed"
    assert task_result.stop_reason == "blocked_without_evidence"
    assert task_result.stop_detail == "我认为可以结束。"
    assert task_result.conclusion == ""
    assert task_result.remaining_gaps


def test_researcher_builds_minimum_result_when_finalization_never_submits():
    loop_state = ResearcherLoopState(active_evidence_limit=2)
    loop_state.add_evidences(
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

    ResearchAgent._apply_minimum_result(loop_state)

    assert loop_state.stop_reason == "fallback_complete"
    assert loop_state.conclusion.startswith("本方向未完成模型综合")
    assert loop_state.remaining_gaps


def test_researcher_builds_blocked_minimum_result_without_evidence():
    loop_state = ResearcherLoopState()

    ResearchAgent._apply_minimum_result(loop_state)

    assert loop_state.stop_reason == "blocked_without_evidence"
    assert loop_state.conclusion == ""
    assert "未获得可用 Evidence。" in loop_state.remaining_gaps


def test_researcher_exhaustion_does_not_promote_unsubmitted_documents_to_evidence():
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

    result = asyncio.run(agent.run(TASK))

    assert result.evidences == []
    assert result.task_result.execution_status == "completed"
    assert result.task_result.stop_reason == "blocked_without_evidence"
    assert result.task_result.conclusion == ""


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

    result = asyncio.run(agent.run(TASK))

    assert result.task_result.stop_reason == "blocked_without_evidence"
    assert result.task_result.queries == ["稳定术语 定义"]
    assert "no_novel_queries" in result.task_result.failures[0]


def test_research_agent_requires_all_dependencies_at_construction():
    with pytest.raises(LLMConfigurationError):
        ResearchAgent(
            None,
            AgentConfig(),
            search_tool=SearchTool(FakeSearchService()),
            reader_tool=FakeReader(),
        )
    with pytest.raises(ValueError, match="SearchTool"):
        ResearchAgent(DirectionLLM([]), AgentConfig(), search_tool=None, reader_tool=FakeReader())


def test_source_reader_requires_material_store_at_construction():
    class Fetcher:
        async def afetch(self, _url, **_kwargs):
            return {"text": "正文"}

    with pytest.raises(TypeError, match="material_store"):
        SourceReaderTool(Fetcher())


def test_source_reader_inlines_short_document_and_registers_it():
    class Fetcher:
        async def afetch(self, _url, **_kwargs):
            return {
                "title": "短文",
                "text": "第一行\n第二行",
                "final_url": "https://example.com/final",
                "retrieval_method": "origin_fetch",
            }

    store = MemoryResearchMaterialStore()
    tool = SourceReaderTool(
        Fetcher(),
        material_store=store,
        document_inline_max_tokens=1_000,
    )
    result = asyncio.run(
        tool.arun(TASK, {"title": "x", "url": "https://example.com", "score": 0.9})
    )

    assert result.status == "completed"
    assert result.documents[0].inline is True
    assert result.documents[0].content == "L1: 第一行\nL2: 第二行"
    assert asyncio.run(store.get(result.documents[0].document_id)).source_url.endswith("/final")


def test_source_reader_keeps_long_document_out_of_tool_result():
    class Fetcher:
        async def afetch(self, _url, **_kwargs):
            return {"title": "长文", "text": "很长的正文" * 100, "final_url": _url}

    tool = SourceReaderTool(
        Fetcher(),
        material_store=MemoryResearchMaterialStore(),
        document_inline_max_tokens=1,
    )
    result = asyncio.run(
        tool.arun(TASK, {"title": "x", "url": "https://example.com", "score": 0.9})
    )

    assert result.status == "completed"
    assert result.documents[0].inline is False
    assert result.documents[0].content == ""
    assert result.documents[0].line_count == 1


def test_researcher_reads_registered_document_and_adds_verified_evidence():
    store = MemoryResearchMaterialStore()
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
        search_tool=SearchTool(FakeSearchService()),
        reader_tool=FakeReader(),
        material_store=store,
    )
    loop_state = ResearcherLoopState(active_evidence_limit=4, evidence_archive_limit=8)
    loop_state.documents[document.document_id] = document

    grep = asyncio.run(
        services.grep_document(
            agent._deps, loop_state, document.document_id, "延迟", 1, 0, "定位数据"
        )
    )
    read = asyncio.run(
        services.read_document(agent._deps, loop_state, document.document_id, [(2, 3)], "读取数据")
    )
    added = asyncio.run(
        services.add_evidence(
            agent._deps,
            TASK,
            loop_state,
            {},
            asyncio.Lock(),
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
        )
    )

    assert grep["match_count"] == 1
    assert "L2:" in read["ranges"][0]["content"]
    assert len(added["accepted"]) == 1
    assert len(added["rejected"]) == 1
    assert loop_state.evidences[0].support == "partial"  # 不得突破来源支撑上限


def test_add_evidence_validates_before_ranking_by_confidence():
    store = MemoryResearchMaterialStore()
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
        search_tool=SearchTool(FakeSearchService()),
        reader_tool=FakeReader(),
        material_store=store,
    )
    loop_state = ResearcherLoopState(active_evidence_limit=2, evidence_archive_limit=2)
    loop_state.documents[document.document_id] = document

    result = asyncio.run(
        services.add_evidence(
            agent._deps,
            TASK,
            loop_state,
            {},
            asyncio.Lock(),
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
        )
    )

    assert len(result["accepted"]) == 1
    assert loop_state.evidences[0].claim == "较高优先级"
    assert result["truncated_submission_count"] == 1
    assert any(
        item["reason"] == "quote_paraphrase" for item in result["rejected"]
    )  # 宽松归一仍不过 → 判定为模型改述


def test_concurrent_add_evidence_commits_under_one_source_limit():
    store = MemoryResearchMaterialStore()
    document = asyncio.run(
        store.put(
            text="事实甲\n事实乙",
            title="并发提交",
            source_url="https://example.com/concurrent",
        )
    )
    agent = ResearchAgent(
        DirectionLLM([]),
        AgentConfig(evidence_add_batch_size=2, evidence_max_per_source=1),
        search_tool=SearchTool(FakeSearchService()),
        reader_tool=FakeReader(),
        material_store=store,
    )
    loop_state = ResearcherLoopState(active_evidence_limit=4, evidence_archive_limit=4)
    loop_state.documents[document.document_id] = document
    commit_lock = asyncio.Lock()

    async def submit(claim: str, quote: str):
        return await services.add_evidence(
            agent._deps,
            TASK,
            loop_state,
            {},
            commit_lock,
            [
                {
                    "document_id": document.document_id,
                    "claim": claim,
                    "quote": quote,
                    "confidence": 0.9,
                }
            ],
            "并发提交",
        )

    async def submit_both():
        return await asyncio.gather(submit("论点甲", "事实甲"), submit("论点乙", "事实乙"))

    results = asyncio.run(submit_both())

    assert sum(len(result["accepted"]) for result in results) == 1
    assert len(loop_state.evidences) == 1
    assert any(
        item["reason"] == "source_evidence_limit_reached"
        for result in results
        for item in result["rejected"]
    )


def test_source_reader_keeps_access_challenge_as_nonfatal_source_outcome():
    class ChallengeFetcher:
        async def afetch(self, _url, **_kwargs):
            raise SourceUnavailableError("access_challenge", "来源返回验证码页")

    tool = SourceReaderTool(ChallengeFetcher(), material_store=MemoryResearchMaterialStore())
    result = asyncio.run(
        tool.arun(TASK, {"title": "x", "url": "https://example.com", "score": 0.9})
    )
    assert result.status == "skipped"
    assert result.reason_code == "access_challenge"


def test_source_reader_registers_tavily_raw_content_when_page_fetch_is_unavailable():
    class BlockedFetcher:
        async def afetch(self, _url, **_kwargs):
            raise SourceUnavailableError("access_challenge", "来源返回验证码页")

    store = MemoryResearchMaterialStore()
    tool = SourceReaderTool(BlockedFetcher(), material_store=store)
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
    ref = asyncio.run(store.get(result.documents[0].document_id))
    assert ref.retrieval_method == "tavily_raw_content"
    assert ref.support_ceiling == "direct"
    assert ref.published_at == "2026-07-04"


def test_source_reader_attaches_search_published_at_to_fetched_document():
    class OkFetcher:
        async def afetch(self, _url, **_kwargs):
            return {"text": "正文", "final_url": "https://example.com"}

    store = MemoryResearchMaterialStore()
    tool = SourceReaderTool(OkFetcher(), material_store=store)
    result = asyncio.run(
        tool.arun(
            TASK,
            {"title": "t", "url": "https://example.com", "published_at": "2026-01-02"},
        )
    )
    assert result.status == "completed"
    ref = asyncio.run(store.get(result.documents[0].document_id))
    assert ref.published_at == "2026-01-02"


def test_source_reader_caps_search_summary_document_at_partial_support():
    class BlockedFetcher:
        async def afetch(self, _url, **_kwargs):
            raise SourceUnavailableError("access_challenge", "来源返回验证码页")

    store = MemoryResearchMaterialStore()
    tool = SourceReaderTool(BlockedFetcher(), material_store=store)
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
    ref = asyncio.run(store.get(result.documents[0].document_id))
    assert ref.retrieval_method == "search_summary"
    assert ref.support_ceiling == "partial"


def test_direction_evidence_pool_releases_slots_without_deleting_archive():
    pool = ResearcherLoopState(active_evidence_limit=2, evidence_archive_limit=4)
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


def test_researcher_agent_crash_passes_through_as_failed_not_dressed_up():
    """如实记录执行状态:图级崩溃必须记 execution_status=failed/direction_agent_failed,
    最小保守结果只改交付(conclusion/gaps),不得把崩溃的方向标成 completed。"""
    agent = researcher_agent(AgentConfig(), decisions=[])

    class ExplodingLoop:
        async def ainvoke(self, *_args, **_kwargs):
            raise RuntimeError("graph exploded")

    agent._agent_loop = ExplodingLoop()
    result = asyncio.run(agent.run(TASK))

    task_result = result.task_result
    assert task_result.execution_status == "failed"
    assert task_result.stop_reason == "direction_agent_failed"
    assert task_result.stop_detail == "graph exploded"
    assert task_result.coverage_status == "insufficient"
    assert any("direction_agent_failed" in item for item in task_result.failures)
    # 交付层仍拿到降级说明与缺口标注(证据为零时的兜底文案不变)
    assert "未获得可用 Evidence。" in task_result.remaining_gaps
