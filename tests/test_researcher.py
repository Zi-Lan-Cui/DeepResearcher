import asyncio
import dataclasses
import hashlib
from types import SimpleNamespace
from typing import Annotated, get_args, get_origin, get_type_hints

import pytest
from langchain_core.messages import AIMessage

from deepresearcher.agents.middleware import StateRebuildMiddleware
from deepresearcher.agents.reducers import merge_active_set, replace_channel, retain_if_absent
from deepresearcher.agents.researcher import Researcher, services
from deepresearcher.agents.researcher.agent import RESEARCHER_SERIAL_TOOLS
from deepresearcher.agents.researcher.state import (
    ResearcherContext,
    ResearcherDeps,
    ResearcherState,
    ResearcherTaskState,
    working_set_snapshot,
)
from deepresearcher.agents.researcher.tools import build_researcher_tools
from deepresearcher.config import AgentConfig
from deepresearcher.evidence.models import Evidence
from deepresearcher.evidence.validator import SourceIndex
from deepresearcher.observability.scope import AgentScope
from deepresearcher.schemas import (
    DocumentReadTarget,
    EvidenceSubmission,
    GrepDocument,
    TaskResult,
)
from deepresearcher.schemas.limits import SEARCH_RESULTS_PREVIEW_COUNT
from deepresearcher.service.execution.checkpoint_serde import build_checkpointer_serde
from deepresearcher.tools import SearchTool, SourceReaderTool
from deepresearcher.tools.errors import SourceUnavailableError, ToolRequestError, UnsafeUrlError
from deepresearcher.tools.web.documents import DocumentRef, DocumentView
from deepresearcher.tools.web.search.models import SearchCandidate
from deepresearcher.vocab import RETRIEVAL_SEARCH_SUMMARY
from fakes import (
    TASK,
    FakeReader,
    FakeSearchService,
    MemoryResearchMaterialStore,
    ResearcherDecision,
    ResearcherLLM,
    make_evidence,
    researcher_agent,
)


def test_task_result_ignores_legacy_answered_points_on_checkpoint_restore() -> None:
    restored = TaskResult.model_validate(
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
        research_agent_max_evidences_per_task=2,
        research_agent_max_turns=4,
        research_agent_max_queries=4,
        research_agent_max_concurrent_source_reads=2,
    )
    reader = CapturingReader()
    agent = researcher_agent(
        config,
        [
            ResearcherDecision(
                action="search",
                reason="尚无证据，先检索定义与直接事实。",
                queries=["稳定术语 定义", "稳定术语 直接证据"],
            ),
            ResearcherDecision(
                action="read",
                reason="读取直接来源。",
                candidate_ids=[
                    "c-"
                    + hashlib.sha1("https://example.com/稳定术语 定义/a".encode()).hexdigest()[:10]
                ],
            ),
            ResearcherDecision(
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
            research_agent_max_evidences_per_task=2,
            research_agent_max_turns=3,
        ),
        [
            ResearcherDecision(action="search", reason="先查直接事实", queries=["方向定义"]),
            ResearcherDecision(
                action="read",
                reason="读取候选来源",
                candidate_ids=[
                    "c-" + hashlib.sha1("https://example.com/方向定义/a".encode()).hexdigest()[:10]
                ],
            ),
            ResearcherDecision(action="complete", reason="材料已足够"),
        ],
    )

    result = asyncio.run(agent.run(TASK))
    assert result.evidences == []
    assert result.task_result.stop_reason == "blocked_without_evidence"
    contents = [
        str(message.content) for snapshot in agent.llm.seen_messages for message in snapshot
    ]
    assert any("委派研究任务" in content for content in contents)
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
    agent = Researcher(
        ResearcherLLM([]),
        AgentConfig(),
        search_tool=SearchTool(client),
        reader_tool=FakeReader(),
        material_store=materials,
    )
    task_state = ResearcherTaskState()
    first = asyncio.run(
        services.search_sources(agent._deps, TASK, task_state, {}, ["分页测试"], "发现候选")
    )

    assert first["result_count"] == 12
    assert len(first["candidates"]) == SEARCH_RESULTS_PREVIEW_COUNT
    assert first["has_more"] is True
    # 预览与分页统一形状：首屏即带截断 snippet，模型无需再调 ListSearchResults(offset=0) 判断。
    assert all(item["snippet"] and len(item["snippet"]) <= 1_200 for item in first["candidates"])
    assert first["next_offset"] == SEARCH_RESULTS_PREVIEW_COUNT  # 预览消费 0..N-1 → 下一页从 N

    page = asyncio.run(
        services.list_search_results(
            agent._deps, TASK, task_state, first["search_id"], 5, 3, "查看下一页"
        )
    )

    assert page["offset"] == 5
    assert page["returned_count"] == 3
    assert page["next_offset"] == 8
    # 同一候选在预览与分页里字段完全一致，不按 include_snippet 分叉。
    assert set(first["candidates"][0]) == set(page["candidates"][0])
    # 分页重建只从 SearchTool 的缓存取数，不请求 Provider。
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
            agent._deps, TASK, ResearcherTaskState(), "search-not-owned", 0, 5, "越权查看"
        )
    )

    assert result == {"status": "rejected", "reason": "unknown_search_id"}


def test_research_agent_can_inspect_and_forget_its_working_set():
    agent = researcher_agent(
        AgentConfig(research_agent_max_evidences_per_task=2, research_agent_max_turns=5),
        [
            ResearcherDecision(action="search", reason="先找来源", queries=["方向"]),
            ResearcherDecision(
                action="read",
                reason="读取来源",
                candidate_ids=[
                    "c-" + hashlib.sha1("https://example.com/方向/a".encode()).hexdigest()[:10]
                ],
            ),
            ResearcherDecision(action="inspect", reason="确认工作集"),
            ResearcherDecision(
                action="release", reason="释放当前材料", evidence_ids=["r1-1-src-unknown-ev-1"]
            ),
            ResearcherDecision(action="complete", reason="停止"),
        ],
    )
    # 该测试只验证工具协议和观察回流；ID 不匹配时 ReleaseEvidence 应安全返回 unknown。
    result = asyncio.run(agent.run(TASK))
    assert result.task_result.execution_status == "completed"
    # 锚在工具回执键而非提示词措辞：模型必须真的看到工作集快照回流。
    assert any(
        "active_evidence" in str(message.content)
        for snapshot in agent.llm.seen_messages
        for message in snapshot
    )


def test_research_agent_can_stop_a_direction_without_unnecessary_search():
    agent = researcher_agent(
        AgentConfig(),
        [
            ResearcherDecision(
                action="complete",
                reason="没有可信的可检索路径。",
            )
        ],
    )

    result = asyncio.run(agent.run(TASK))

    assert result.task_result.execution_status == "completed"
    assert result.task_result.stop_reason == "blocked_without_evidence"
    # 无证据时不采纳模型自述，系统把停止原因写成对当前状态的如实说明。
    assert "没有可信的可检索路径" in result.task_result.conclusion


def test_research_agent_converts_completion_without_evidence_to_blocked_result():
    agent = researcher_agent(
        AgentConfig(),
        [
            ResearcherDecision(
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
    assert task_result.conclusion.startswith("本任务未取得可锚定的 Evidence")


def test_researcher_builds_minimum_result_when_finalization_never_submits():
    task_state = ResearcherTaskState(active_evidence_limit=2)
    task_state.add_evidences(
        [
            Evidence(
                evidence_id="e1",
                task_id="t1",
                research_direction="方向",
                claim="已验证的有限事实",
                quote="原文片段",
                source_url="https://example.com/a",
                support="strict",
            )
        ]
    )

    Researcher._apply_minimum_result(task_state)

    assert task_state.stop_reason == "fallback_complete"
    assert task_state.conclusion.startswith("本任务未在回合限制内提交结构化总结")
    assert "综合结论尚未形成" in task_state.conclusion


def test_researcher_builds_blocked_minimum_result_without_evidence():
    task_state = ResearcherTaskState()

    Researcher._apply_minimum_result(task_state)

    assert task_state.stop_reason == "blocked_without_evidence"
    assert "没有取得可锚定的 Evidence" in task_state.conclusion


def test_researcher_exhaustion_does_not_promote_unsubmitted_documents_to_evidence():
    agent = researcher_agent(
        AgentConfig(
            research_agent_max_turns=1,
            submission_max_rejections=1,
            research_agent_max_evidences_per_task=2,
        ),
        [
            ResearcherDecision(action="search", reason="先找来源", queries=["方向"]),
            ResearcherDecision(
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
    # 未提交的文档不得升格为证据；conclusion 只留系统的状态说明，没有研究结论。
    assert "没有取得可锚定的 Evidence" in result.task_result.conclusion


def test_research_direction_decision_rejects_conclusions_during_search():
    with pytest.raises(ValueError, match="只有 action=complete"):
        ResearcherDecision(
            action="search",
            reason="还要继续检索。",
            queries=["新检索式"],
            conclusion="不应在搜索阶段给出结论。",
        )


def test_research_agent_replans_duplicate_queries_instead_of_mislabeling_budget_exhaustion():
    config = AgentConfig(
        research_agent_max_turns=3,
        research_agent_max_queries=4,
        research_agent_max_evidences_per_task=4,
    )
    agent = researcher_agent(
        config,
        [
            ResearcherDecision(action="search", reason="先检索", queries=["稳定术语 定义"]),
            ResearcherDecision(action="search", reason="误重复旧检索式", queries=["稳定术语 定义"]),
            ResearcherDecision(action="complete", reason="现有材料足以谨慎回答。"),
        ],
    )

    result = asyncio.run(agent.run(TASK))

    assert result.task_result.stop_reason == "blocked_without_evidence"
    assert result.task_result.queries == ["稳定术语 定义"]
    assert "no_novel_queries" in result.task_result.failures[0]


def test_research_agent_names_query_quota_exhaustion_instead_of_duplication():
    config = AgentConfig(
        research_agent_max_turns=3,
        research_agent_max_queries=1,
        research_agent_max_evidences_per_task=4,
    )
    agent = researcher_agent(
        config,
        [
            ResearcherDecision(action="search", reason="先检索", queries=["稳定术语 定义"]),
            ResearcherDecision(
                action="search", reason="换个全新检索式", queries=["另一组全新检索词"]
            ),
            ResearcherDecision(action="complete", reason="现有材料足以谨慎回答。"),
        ],
    )

    result = asyncio.run(agent.run(TASK))

    # 查询是新的，失败原因必须写成额度而不是重复。
    assert result.task_result.queries == ["稳定术语 定义"]
    assert any(item.startswith("query_quota_exhausted") for item in result.task_result.failures)
    assert not any(item.startswith("no_novel_queries") for item in result.task_result.failures)


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
    agent = Researcher(
        ResearcherLLM([]),
        AgentConfig(evidence_max_per_source=4),
        search_tool=SearchTool(FakeSearchService()),
        reader_tool=FakeReader(),
        material_store=store,
    )
    task_state = ResearcherTaskState(active_evidence_limit=4, evidence_archive_limit=8)
    task_state.documents[document.document_id] = document

    grep = asyncio.run(
        services.grep_document(
            agent._deps, task_state, document.document_id, ["延迟"], [(0, 6)], 1, "定位数据"
        )
    )
    read = asyncio.run(
        services.read_document(agent._deps, task_state, document.document_id, [(2, 3)], "读取数据")
    )
    added = asyncio.run(
        services.add_evidence(
            agent._deps,
            TASK,
            task_state,
            {},
            [
                {
                    "document_id": document.document_id,
                    "claim": "实验端到端延迟为 20ms。",
                    "quote_start": "实验表明端到端延迟",
                    "quote_end": "20ms",
                    "support": "strict",
                    "confidence": 0.9,
                },
                {
                    "document_id": document.document_id,
                    "claim": "不存在的事实",
                    "quote_start": "正文里没有这句话",
                    "support": "strict",
                    "confidence": 0.9,
                },
            ],
            "批量提交",
        )
    )

    assert grep["pattern_totals"] == {"延迟": 1}
    assert grep["segments"][0]["returned"] == 1
    assert "L2:" in read["ranges"][0]["content"]
    assert len(added["accepted"]) == 1
    assert len(added["rejected"]) == 1
    assert task_state.evidences[0].support == "partial"  # 不得突破来源支撑上限


def test_document_id_sanitizer_strips_injected_control_chars():
    contaminated = "doc\r-1b41c8c5\n"
    target = DocumentReadTarget.model_validate(
        {"document_id": contaminated, "ranges": [{"start_line": 2, "end_line": 3}]}
    )
    assert target.document_id == "doc-1b41c8c5"
    submission = EvidenceSubmission.model_validate(
        {
            "document_id": contaminated,
            "claim": "主张",
            "quote_start": "开头",
            "support": "strict",
            "confidence": 0.9,
        }
    )
    assert submission.document_id == "doc-1b41c8c5"
    grep_args = GrepDocument.model_validate(
        {"document_id": contaminated, "patterns": ["词"], "reason": "定位"}
    )
    assert grep_args.document_id == "doc-1b41c8c5"
    assert grep_args.segments[0].offset == 0  # segments 缺省即从流首读取


def test_read_documents_batches_several_documents_into_one_call():
    store = MemoryResearchMaterialStore()
    first = asyncio.run(
        store.put(
            text="引言\n甲文延迟为 12ms。\n结论", title="甲报告", source_url="https://example.com/a"
        )
    )
    second = asyncio.run(
        store.put(
            text="引言\n乙文延迟为 34ms。\n结论", title="乙报告", source_url="https://example.com/b"
        )
    )
    agent = Researcher(
        ResearcherLLM([]),
        AgentConfig(),
        search_tool=SearchTool(FakeSearchService()),
        reader_tool=FakeReader(),
        material_store=store,
    )
    task_state = ResearcherTaskState(active_evidence_limit=4, evidence_archive_limit=8)
    task_state.documents[first.document_id] = first
    task_state.documents[second.document_id] = second

    batch = asyncio.run(
        services.read_documents(
            agent._deps,  # noqa: SLF001
            task_state,
            [(first.document_id, [(2, 2)]), (second.document_id, [(2, 2)])],
            "跨文档批读",
        )
    )
    assert batch["status"] == "completed"
    documents = batch["documents"]
    assert isinstance(documents, list) and len(documents) == 2
    assert [item["document_id"] for item in documents] == [first.document_id, second.document_id]
    assert "12ms" in documents[0]["ranges"][0]["content"]
    assert "34ms" in documents[1]["ranges"][0]["content"]

    mixed_unknown = asyncio.run(
        services.read_documents(
            agent._deps,  # noqa: SLF001
            task_state,
            [(first.document_id, [(2, 2)]), ("doc-未登记", [(1, 1)])],
            "混入未登记文档",
        )
    )
    assert mixed_unknown["status"] == "rejected"
    assert mixed_unknown["reason"] == "unknown_document_id"


def test_working_set_snapshot_carries_direction_and_source_identity():
    evidence = Evidence(
        evidence_id="ev-1",
        task_id="task-0001",
        research_direction="方向甲",
        claim="主张",
        quote="原文",
        source_url="https://example.com/x",
        source_title="来源页",
        support="strict",
        confidence=0.8,
    )
    task_state = ResearcherTaskState(active_evidence_limit=4, evidence_archive_limit=8)
    task_state.evidences = [evidence]
    task_state.active_evidence_ids = {"ev-1"}

    item = working_set_snapshot(task_state)["active_evidence"][0]
    assert item["task_id"] == "task-0001"
    assert item["source_title"] == "来源页"
    assert item["source_domain"] == "example.com"


def test_add_evidence_validates_before_ranking_by_confidence():
    store = MemoryResearchMaterialStore()
    document = asyncio.run(
        store.put(
            text="低优先级事实\n高优先级事实",
            title="排序测试",
            source_url="https://example.com/ranking",
        )
    )
    agent = Researcher(
        ResearcherLLM([]),
        AgentConfig(evidence_add_batch_size=1, evidence_max_per_source=2),
        search_tool=SearchTool(FakeSearchService()),
        reader_tool=FakeReader(),
        material_store=store,
    )
    task_state = ResearcherTaskState(active_evidence_limit=2, evidence_archive_limit=2)
    task_state.documents[document.document_id] = document

    result = asyncio.run(
        services.add_evidence(
            agent._deps,
            TASK,
            task_state,
            {},
            [
                {
                    "document_id": document.document_id,
                    "claim": "无效但最高分",
                    "quote_start": "并不存在",
                    "confidence": 1.0,
                },
                {
                    "document_id": document.document_id,
                    "claim": "较低优先级",
                    "quote_start": "低优先级事实",
                    "confidence": 0.2,
                },
                {
                    "document_id": document.document_id,
                    "claim": "较高优先级",
                    "quote_start": "高优先级事实",
                    "confidence": 0.9,
                },
            ],
            "验证后排序",
        )
    )

    assert len(result["accepted"]) == 1
    assert task_state.evidences[0].claim == "较高优先级"
    assert result["truncated_submission_count"] == 1
    assert any(
        item["reason"] == "anchor_not_found" for item in result["rejected"]
    )  # 锚未命中的高置信提交在排序前就被剔除


def test_add_evidence_anchor_resolves_across_punctuation_style():
    # 模型只交词锚，弯引号/逗号样式由程序回填的原文子串承担，模型不需要抄对这些符号。
    store = MemoryResearchMaterialStore()
    document = asyncio.run(
        store.put(
            text="报告称“非常稳健”，通过验收。其余正文。",
            title="标点修复",
            source_url="https://example.com/repair",
        )
    )
    agent = Researcher(
        ResearcherLLM([]),
        AgentConfig(),
        search_tool=SearchTool(FakeSearchService()),
        reader_tool=FakeReader(),
        material_store=store,
    )
    task_state = ResearcherTaskState(active_evidence_limit=4, evidence_archive_limit=4)
    task_state.documents[document.document_id] = document
    events: list[dict[str, object]] = []

    def capture_emit(event_type: str, payload: dict[str, object], **kwargs: object) -> None:
        del kwargs
        if event_type == "direction_evidence_added":
            events.append(payload)

    deps = dataclasses.replace(agent._deps, emit=capture_emit)
    result = asyncio.run(
        services.add_evidence(
            deps,
            TASK,
            task_state,
            {},
            [
                {
                    "document_id": document.document_id,
                    "claim": "系统被认定稳健。",
                    # 首尾锚只取两端词，中间整段(含弯引号)由程序逐字回填。
                    "quote_start": "报告称",
                    "quote_end": "通过验收",
                    "confidence": 0.9,
                }
            ],
            "锚点提交",
        )
    )

    assert len(result["accepted"]) == 1
    assert task_state.evidences[0].quote == "报告称“非常稳健”，通过验收"
    assert events[0]["accepted_count"] == 1
    assert events[0]["rejected_count"] == 0


def test_search_event_payload_carries_tool_call_id():
    # 同一任务可多次检索。载荷缺 tool_call_id 时各次去重键相同，事件存储只保留首次，
    # 前端只收到第一轮的检索行。
    events: list[dict[str, object]] = []

    def capture_emit(event_type: str, payload: dict[str, object], **kwargs: object) -> None:
        del kwargs
        if event_type == "direction_search_completed":
            events.append(payload)

    deps = ResearcherDeps(
        config=AgentConfig(),
        search_tool=SearchTool(FakeSearchService()),
        reader_tool=FakeReader(),
        material_store=MemoryResearchMaterialStore(),
        emit=capture_emit,
    )
    context = ResearcherContext(
        deps=deps,
        scope=AgentScope.from_task(TASK, agent_name="Researcher"),
        task=TASK,
        task_state=ResearcherTaskState(active_evidence_limit=4, evidence_archive_limit=4),
        event_context={"task_id": TASK["id"], "operation_id": "research-task-0001"},
    )
    runtime = SimpleNamespace(context=context, tool_call_id="call-search-1")
    tool = {item.name: item for item in build_researcher_tools()}["SearchSources"]

    asyncio.run(tool.coroutine(reason="查找快照机制", queries=["Redis RDB"], runtime=runtime))

    assert events[0]["queries"] == ["Redis RDB"]
    assert events[0]["tool_call_id"] == "call-search-1"


def test_add_evidence_event_payload_carries_tool_call_id():
    # 同一任务可多次提交证据。载荷缺 tool_call_id 时各次去重键相同，事件存储会把后几次
    # 判为重放丢弃，前端只收到首次提交。
    store = MemoryResearchMaterialStore()
    document = asyncio.run(
        store.put(
            text="Redis 的 RDB 会在指定时间间隔内生成数据集快照。",
            title="快照机制",
            source_url="https://example.com/rdb",
        )
    )
    task_state = ResearcherTaskState(active_evidence_limit=4, evidence_archive_limit=4)
    task_state.documents[document.document_id] = document
    events: list[dict[str, object]] = []

    def capture_emit(event_type: str, payload: dict[str, object], **kwargs: object) -> None:
        del kwargs
        if event_type == "direction_evidence_added":
            events.append(payload)

    deps = ResearcherDeps(
        config=AgentConfig(),
        search_tool=SearchTool(FakeSearchService()),
        reader_tool=FakeReader(),
        material_store=store,
        emit=capture_emit,
    )
    context = ResearcherContext(
        deps=deps,
        scope=AgentScope.from_task(TASK, agent_name="Researcher"),
        task=TASK,
        task_state=task_state,
        event_context={"task_id": TASK["id"], "operation_id": "research-task-0001"},
    )
    runtime = SimpleNamespace(context=context, tool_call_id="call-evidence-1")
    tool = {item.name: item for item in build_researcher_tools()}["AddEvidence"]

    asyncio.run(
        tool.coroutine(
            evidences=[
                {
                    "document_id": document.document_id,
                    "claim": "RDB 按时间间隔生成快照。",
                    "quote_start": "RDB",
                    "quote_end": "数据集快照",
                    "confidence": 0.8,
                }
            ],
            reason="提交快照机制的原文",
            runtime=runtime,
        )
    )

    assert events[0]["accepted_count"] == 1
    assert events[0]["tool_call_id"] == "call-evidence-1"


def test_read_document_reports_line_count_when_ranges_exceed_document():
    # 行段整体越过文档末尾时 ranges 为空且状态仍是 completed；回执带实际行数，
    # 模型才能判断该改读哪一段，而不是重复同一请求。
    store = MemoryResearchMaterialStore()
    document = asyncio.run(
        store.put(
            text="第一行\n第二行\n第三行", title="三行", source_url="https://example.com/three"
        )
    )
    task_state = ResearcherTaskState(active_evidence_limit=4, evidence_archive_limit=4)
    task_state.documents[document.document_id] = document
    deps = ResearcherDeps(
        config=AgentConfig(),
        search_tool=SearchTool(FakeSearchService()),
        reader_tool=FakeReader(),
        material_store=store,
        emit=lambda *_args, **_kwargs: None,
    )

    read = asyncio.run(
        services.read_document(deps, task_state, document.document_id, [(9, 12)], "越界行段")
    )

    assert read["status"] == "completed"
    assert read["ranges"] == []
    assert read["line_count"] == 3


def _state_with_one_active_evidence() -> ResearcherTaskState:
    task_state = ResearcherTaskState(active_evidence_limit=4, evidence_archive_limit=4)
    task_state.evidences.append(
        Evidence(
            evidence_id="ev-1",
            task_id="task-0001",
            research_direction="方向甲",
            claim="RDB 按间隔生成快照",
            quote="保存数据集快照",
            source_url="https://example.com/rdb",
            support="strict",
        )
    )
    task_state.active_evidence_ids = {"ev-1"}
    return task_state


def test_complete_task_writes_finding_alongside_conclusion():
    # finding 与 conclusion 由同一次提交写入，各走一条通道：前者上卡面，后者给 Supervisor。
    task_state = _state_with_one_active_evidence()

    result = services.complete_task(
        task_state, "取证足够", ["ev-1"], "完整叙述。", "RDB 按间隔做全量快照。"
    )

    assert result["status"] == "accepted"
    assert task_state.finding == "RDB 按间隔做全量快照。"
    assert task_state.conclusion == "完整叙述。"


def test_complete_task_clears_finding_when_nothing_is_selected():
    # 无活跃证据时模型自述无从校验，摘要与结论一并置空，卡面不出现摘要行。
    task_state = ResearcherTaskState(active_evidence_limit=4, evidence_archive_limit=4)

    result = services.complete_task(task_state, "没查到", [], "模型自述。", "模型自述摘要。")

    assert result["status"] == "accepted"
    assert task_state.finding == ""
    assert "未取得可锚定的 Evidence" in task_state.conclusion


def test_task_complete_tool_puts_finding_into_the_channel():
    # 摘要经通道回传，而不是由前端截取 conclusion：两者长度与语气都不一样。
    task_state = _state_with_one_active_evidence()
    deps = ResearcherDeps(
        config=AgentConfig(),
        search_tool=SearchTool(FakeSearchService()),
        reader_tool=FakeReader(),
        material_store=None,
        emit=lambda *_args, **_kwargs: None,
    )
    context = ResearcherContext(
        deps=deps,
        scope=AgentScope.from_task(TASK, agent_name="Researcher"),
        task=TASK,
        task_state=task_state,
        event_context={"task_id": TASK["id"], "operation_id": "research-task-0001"},
    )
    runtime = SimpleNamespace(context=context, tool_call_id="call-complete-1")
    tool = {item.name: item for item in build_researcher_tools()}["TaskComplete"]

    command = asyncio.run(
        tool.coroutine(
            reason="取证足够",
            selected_evidence_ids=["ev-1"],
            conclusion="完整叙述。",
            finding="RDB 按间隔做全量快照。",
            runtime=runtime,
        )
    )

    assert command.update["finding"] == "RDB 按间隔做全量快照。"
    assert command.update["conclusion"] == "完整叙述。"


def test_add_evidence_scans_each_document_once(monkeypatch):
    """同一文档的多条锚点共用一次全文归一：那是这条路径上唯一的 CPU 热点。"""
    scans: list[str] = []
    original_scan = SourceIndex.scan.__func__

    def counting_scan(cls, text):
        scans.append(text)
        return original_scan(cls, text)

    monkeypatch.setattr(SourceIndex, "scan", classmethod(counting_scan))

    store = MemoryResearchMaterialStore()
    document = asyncio.run(
        store.put(
            text="事实甲在此。事实乙在此。事实丙在此。",
            title="一次归一",
            source_url="https://example.com/once",
        )
    )
    agent = Researcher(
        ResearcherLLM([]),
        AgentConfig(evidence_max_per_source=8),
        search_tool=SearchTool(FakeSearchService()),
        reader_tool=FakeReader(),
        material_store=store,
    )
    task_state = ResearcherTaskState(active_evidence_limit=8, evidence_archive_limit=8)
    task_state.documents[document.document_id] = document

    result = asyncio.run(
        services.add_evidence(
            agent._deps,
            TASK,
            task_state,
            {},
            [
                {
                    "document_id": document.document_id,
                    "claim": f"{name}成立。",
                    "quote_start": f"事实{name}",
                    "confidence": 0.9,
                }
                for name in ("甲", "乙", "丙")
            ],
            "同文档多锚点",
        )
    )

    assert len(result["accepted"]) == 3
    assert scans == ["事实甲在此。事实乙在此。事实丙在此。"]


def test_add_evidence_anchor_miss_receipt_carries_nearest_source_sentence():
    store = MemoryResearchMaterialStore()
    document = asyncio.run(
        store.put(
            text="The system remained stable for 50 hours. Another matter entirely here.",
            title="锚未命中提示",
            source_url="https://example.com/hint",
        )
    )
    agent = Researcher(
        ResearcherLLM([]),
        AgentConfig(),
        search_tool=SearchTool(FakeSearchService()),
        reader_tool=FakeReader(),
        material_store=store,
    )
    task_state = ResearcherTaskState(active_evidence_limit=4, evidence_archive_limit=4)
    task_state.documents[document.document_id] = document

    result = asyncio.run(
        services.add_evidence(
            agent._deps,
            TASK,
            task_state,
            {},
            [
                {
                    "document_id": document.document_id,
                    "claim": "系统稳定运行两天。",
                    "quote_start": "The system stayed up for two days",
                    "confidence": 0.9,
                }
            ],
            "锚未命中提交",
        )
    )

    assert result["rejected"][0]["reason"] == "anchor_not_found"
    assert result["rejected"][0]["nearby_original_text"] == (
        "The system remained stable for 50 hours"
    )


def test_concurrent_add_evidence_commits_under_one_source_limit():
    store = MemoryResearchMaterialStore()
    document = asyncio.run(
        store.put(
            text="事实甲\n事实乙",
            title="并发提交",
            source_url="https://example.com/concurrent",
        )
    )
    agent = Researcher(
        ResearcherLLM([]),
        AgentConfig(evidence_add_batch_size=2, evidence_max_per_source=1),
        search_tool=SearchTool(FakeSearchService()),
        reader_tool=FakeReader(),
        material_store=store,
    )
    task_state = ResearcherTaskState(active_evidence_limit=4, evidence_archive_limit=4)
    task_state.documents[document.document_id] = document

    async def submit(claim: str, quote_start: str):
        return await services.add_evidence(
            agent._deps,
            TASK,
            task_state,
            {},
            [
                {
                    "document_id": document.document_id,
                    "claim": claim,
                    "quote_start": quote_start,
                    "confidence": 0.9,
                }
            ],
            "并发提交",
        )

    async def submit_both():
        return await asyncio.gather(submit("论点甲", "事实甲"), submit("论点乙", "事实乙"))

    results = asyncio.run(submit_both())

    assert sum(len(result["accepted"]) for result in results) == 1
    assert len(task_state.evidences) == 1
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
    assert ref.support_ceiling == "strict"
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
    pool = ResearcherTaskState(active_evidence_limit=2, evidence_archive_limit=4)
    first = [
        Evidence(
            evidence_id=f"e{index}",
            task_id="task-1",
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
    # 交付层仍拿到降级说明，证据为零时的默认文案不变
    assert "没有取得可锚定的 Evidence" in task_result.conclusion


def test_search_tool_emits_query_level_progress_events():
    """方向卡的无输出间隙由逐 query 事件填充:每条检索式起止各一条,聚合事件保留。"""

    class Sink:
        def __init__(self) -> None:
            self.records = []

        def write(self, record) -> None:
            self.records.append(record)

    sink = Sink()
    tool = SearchTool(FakeSearchService(), event_sink=sink)
    asyncio.run(tool.arun_queries(TASK, queries=["甲", "乙"]))

    query_events = {
        (record.event_type, record.payload["query"])
        for record in sink.records
        if record.event_type.startswith("search_query_")
    }
    assert {("search_query_started", "甲"), ("search_query_completed", "甲")} <= query_events
    assert {("search_query_started", "乙"), ("search_query_completed", "乙")} <= query_events
    # 整批聚合的 tool_started/tool_completed 仍在：卡的首行与批次行不受影响。
    assert {r.event_type for r in sink.records} >= {"tool_started", "tool_completed"}


def test_search_tool_query_failed_event_reraises():
    class BoomClient:
        provider_name = "boom"

        async def asearch(self, _query):
            raise RuntimeError("上游 5xx")

    class Sink:
        def __init__(self) -> None:
            self.records = []

        def write(self, record) -> None:
            self.records.append(record)

    sink = Sink()
    tool = SearchTool(BoomClient(), event_sink=sink)
    result = asyncio.run(tool.arun_queries(TASK, queries=["会失败"]))
    assert result.status == "failed"
    assert any(
        r.event_type == "search_query_failed" and r.payload["query"] == "会失败"
        for r in sink.records
    )


# 通道化不变式守卫：覆盖 §5 硬点、通道往返、ref 化、批重建与 serde。


def _non_concurrency_safe_channels(state_cls) -> set[str]:
    """replace / retain-if-absent 语义的通道:整字段快照,只许 serial 工具写。"""
    hints = get_type_hints(state_cls, include_extras=True)
    unsafe = {replace_channel, retain_if_absent}
    result = set()
    for name, hint in hints.items():
        if get_origin(hint) is Annotated:
            reducers = [arg for arg in get_args(hint)[1:] if callable(arg)]
            if any(reducer in unsafe for reducer in reducers):
                result.add(name)
    return result


def test_researcher_concurrent_tools_are_known_merge_writers():
    """§5 分类守卫:非串行工具恰为已知的合并语义写者;新增并发工具必须显式归类。"""
    names = {item.name for item in build_researcher_tools()}
    concurrent = names - set(RESEARCHER_SERIAL_TOOLS)
    assert concurrent == {
        "AddEvidence",
        "ReadSources",
        "SearchSources",
        "ListSearchResults",
        "GrepDocument",
        "ReadDocument",
    }


def test_researcher_selection_channels_are_serial_only():
    """非并发安全通道 = 方向收尾(TaskComplete,串行)独写集。"""
    assert _non_concurrency_safe_channels(ResearcherState) == {
        "conclusion",
        "finding",
        "stop_reason",
        "stop_detail",
    }


def test_researcher_channel_round_trip_strips_bodies():
    """initial_channel_state → from_channel_values 等价;候选去 raw_content、文档降 DocumentRef。"""
    state = ResearcherTaskState(active_evidence_limit=6, evidence_archive_limit=12)
    state.candidates["c-1"] = SearchCandidate(
        candidate_id="c-1",
        url="https://a",
        title="标题",
        snippet="摘要",
        raw_content="这是一大段不应进通道的正文",
        score=1.0,
    )
    state.documents["d-1"] = DocumentView(
        document_id="d-1",
        source_url="https://a",
        title="标题",
        content_hash="h",
        line_count=3,
        token_count=10,
        inline=True,
        content="文档正文也不进通道",
    )
    state.add_evidences([make_evidence("e1", "事实")])
    state.selected_candidate_ids.add("c-1")
    state.read_urls.append("https://a")

    channels = state.initial_channel_state()
    # ref 化纪律：通道里既无候选正文，也无文档正文。
    assert channels["candidates"]["c-1"].raw_content == ""
    assert "content" not in channels["documents"]["d-1"].model_dump()
    assert isinstance(channels["documents"]["d-1"], DocumentRef)

    # 入口 input 在生产里经 langgraph 施加通道 reducer；测试如实模拟这一步。
    channels["active_evidence_ids"] = merge_active_set(None, channels["active_evidence_ids"])
    restored = ResearcherTaskState.from_channel_values(
        channels, active_evidence_limit=6, evidence_archive_limit=12
    )
    assert restored.candidates["c-1"].url == "https://a"
    assert restored.candidates["c-1"].snippet == "摘要"
    assert set(restored.documents) == {"d-1"}
    assert restored.documents["d-1"].source_url == "https://a"
    assert restored.selected_candidate_ids == {"c-1"}
    assert [item.evidence_id for item in restored.evidences] == ["e1"]
    assert restored.active_evidence_ids == state.active_evidence_ids
    assert restored.read_urls == ["https://a"]


def test_researcher_rebuild_discards_uncommitted_memory():
    """每模型回合从通道重建:未经 Command 提交的内存改动被丢弃。"""

    class _Runtime:
        def __init__(self, context):
            self.context = context

    committed = ResearcherTaskState(active_evidence_limit=6, evidence_archive_limit=12)
    committed.conclusion = "已提交结论"
    channels = committed.initial_channel_state()
    # 入口 input 在生产里经 langgraph 施加通道 reducer；测试如实模拟这一步。
    channels["active_evidence_ids"] = merge_active_set(None, channels["active_evidence_ids"])
    context = ResearcherContext(
        deps=SimpleNamespace(config=AgentConfig()),
        scope=SimpleNamespace(),
        task=TASK,
        task_state=committed,
    )
    context.task_state.conclusion = "未提交的内存改动"

    asyncio.run(StateRebuildMiddleware().abefore_model(channels, _Runtime(context)))
    assert context.task_state.conclusion == "已提交结论"


def test_researcher_candidate_channel_survives_msgpack():
    """进通道的候选/文档 ref 形经检查点 serde 往返,类型受白名单放行、正文保持剥离。"""
    serde = build_checkpointer_serde()
    payload = {
        "candidates": {
            "c-1": SearchCandidate(candidate_id="c-1", url="https://a", snippet="摘要", score=1.0)
        },
        "documents": {
            "d-1": DocumentRef(
                document_id="d-1",
                source_url="https://a",
                content_hash="h",
                line_count=1,
                token_count=1,
            )
        },
    }
    restored = serde.loads_typed(serde.dumps_typed(payload))
    assert isinstance(restored["candidates"]["c-1"], SearchCandidate)
    assert isinstance(restored["documents"]["d-1"], DocumentRef)
    assert restored["candidates"]["c-1"] == payload["candidates"]["c-1"]


def test_unknown_id_receipts_carry_available_lists():
    """三类 unknown document_id 驳回都附有效清单，供模型改引用。"""
    store = MemoryResearchMaterialStore()
    document = asyncio.run(
        store.put(text="原文一句。", title="文档", source_url="https://example.com/list")
    )
    agent = Researcher(
        ResearcherLLM([]),
        AgentConfig(),
        search_tool=SearchTool(FakeSearchService()),
        reader_tool=FakeReader(),
        material_store=store,
    )
    task_state = ResearcherTaskState(active_evidence_limit=4, evidence_archive_limit=4)
    task_state.documents[document.document_id] = document

    added = asyncio.run(
        services.add_evidence(
            agent._deps,  # noqa: SLF001
            TASK,
            task_state,
            {},
            [
                {
                    "document_id": "doc-hallucinated",
                    "claim": "c",
                    "quote_start": "q",
                    "confidence": 0.5,
                }
            ],
            "幻觉提交",
        )
    )
    assert added["rejected"][0]["reason"] == "unknown_document_id"
    assert added["available_document_ids"] == [document.document_id]

    grep = asyncio.run(
        services.grep_document(
            agent._deps, task_state, "doc-hallucinated", ["原文"], [(0, 6)], 2, "r"
        )  # noqa: SLF001
    )
    assert grep["available_document_ids"] == [document.document_id]
    read = asyncio.run(
        services.read_document(agent._deps, task_state, "doc-hallucinated", [(1, 2)], "r")  # noqa: SLF001
    )
    assert read["available_document_ids"] == [document.document_id]


def test_read_sources_unknown_candidates_list_available_ids():
    """读取候选全不存在时，回执附可用候选 id；无候选时给空列表也如实可见。"""
    store = MemoryResearchMaterialStore()
    agent = Researcher(
        ResearcherLLM([]),
        AgentConfig(),
        search_tool=SearchTool(FakeSearchService()),
        reader_tool=FakeReader(),
        material_store=store,
    )
    task_state = ResearcherTaskState()
    task_state.candidates["c-real"] = SearchCandidate(
        candidate_id="c-real", url="https://example.com/real", title="真实候选"
    )

    receipt = asyncio.run(
        services.read_sources(
            agent._deps,
            TASK,
            task_state,
            {},
            ["c-hallucinated"],
            "幻觉候选",  # noqa: SLF001
        )
    )
    assert receipt["unknown_candidate_ids"] == ["c-hallucinated"]
    assert receipt["available_candidate_ids"] == ["c-real"]


def test_read_sources_names_search_summary_documents_as_non_primary():
    """回退摘要登记成文档时要标为非原文。"""

    class SummaryReader:
        async def arun(self, task, candidate):
            return {
                "task_id": task["id"],
                "status": "completed",
                "source_url": candidate["url"],
                "documents": [
                    {
                        "document_id": "doc-summary",
                        "source_url": candidate["url"],
                        "title": "订阅墙章节",
                        "content_hash": "h1",
                        "line_count": 2,
                        "token_count": 18,
                        "retrieval_method": RETRIEVAL_SEARCH_SUMMARY,
                        "support_ceiling": "partial",
                    }
                ],
            }

    agent = Researcher(
        ResearcherLLM([]),
        AgentConfig(),
        search_tool=SearchTool(FakeSearchService()),
        reader_tool=SummaryReader(),
        material_store=MemoryResearchMaterialStore(),
    )
    task_state = ResearcherTaskState()
    task_state.candidates["c-1"] = SearchCandidate(
        candidate_id="c-1", url="https://paid.example/ch04", title="订阅墙章节"
    )

    receipt = asyncio.run(
        services.read_sources(agent._deps, TASK, task_state, {}, ["c-1"], "读订阅墙")  # noqa: SLF001
    )

    notices = receipt.get("document_notices") or []
    assert len(notices) == 1
    assert "不能当原文引用" in str(notices[0])
    assert RETRIEVAL_SEARCH_SUMMARY in str(notices[0])


def test_fetch_guarded_fallback_emits_url_bearing_audit_event():
    """SSRF 守卫拦截后由搜索内容顶替登记时，落一条带 requested_url 与归因的审计事件。"""

    class Sink:
        def __init__(self) -> None:
            self.records = []

        def write(self, record) -> None:
            self.records.append(record)

    class GuardedFetcher:
        async def afetch(self, _url, **_kwargs):
            # 与真实守卫一致：命中拦截段时显式标 ssrf_block。
            raise UnsafeUrlError("出于安全原因，不能访问本机、私网或保留地址。", "ssrf_block")

    store = MemoryResearchMaterialStore()
    sink = Sink()
    tool = SourceReaderTool(GuardedFetcher(), material_store=store, event_sink=sink)
    result = asyncio.run(
        tool.arun(
            TASK,
            {
                "title": "来源",
                "url": "http://127.0.0.1/private",
                "raw_content": "提供方已带回的正文。",
                "snippet": "",
                "content_provider": "tavily",
            },
        )
    )

    assert result.status == "completed"  # 回退路径照常登记，模型侧不报错
    fallbacks = [r for r in sink.records if r.event_type == "source_read_fallback"]
    assert len(fallbacks) == 1
    payload = fallbacks[0].payload
    assert payload["reason_code"] == "ssrf_block"
    assert payload["requested_url"] == "http://127.0.0.1/private"
    assert payload["fallback_retrieval_method"] == "tavily_raw_content"


def test_fetch_http_403_is_skipped_as_access_challenge():
    class Sink:
        def __init__(self) -> None:
            self.records = []

        def write(self, record) -> None:
            self.records.append(record)

    class RejectedFetcher:
        async def afetch(self, _url, **_kwargs):
            error = ToolRequestError("HTTP 403 from url", retryable=False)
            error.status_code = 403
            raise error

    sink = Sink()
    tool = SourceReaderTool(
        RejectedFetcher(), material_store=MemoryResearchMaterialStore(), event_sink=sink
    )
    result = asyncio.run(
        tool.arun(
            TASK,
            {
                "title": "订阅墙章节",
                "url": "https://paid.example/ch04",
                "snippet": "",
                "raw_content": "",
            },
        )
    )

    # 订阅墙是来源级结论，不是网络故障：标 skipped，模型才不会反复重试。
    assert result.status == "skipped"
    assert result.reason_code == "access_challenge"


def test_fetch_dns_failure_is_not_mislabeled_as_ssrf_block():
    class Sink:
        def __init__(self) -> None:
            self.records = []

        def write(self, record) -> None:
            self.records.append(record)

    class ResolveFailedFetcher:
        async def afetch(self, _url, **_kwargs):
            raise UnsafeUrlError("URL 主机名无法解析（DNS 查询失败：11001）。", "resolve_failed")

    sink = Sink()
    tool = SourceReaderTool(
        ResolveFailedFetcher(), material_store=MemoryResearchMaterialStore(), event_sink=sink
    )
    result = asyncio.run(
        tool.arun(
            TASK,
            {
                "title": "来源",
                "url": "https://flaky.example/a",
                "snippet": "",
                "raw_content": "",
            },
        )
    )

    assert result.status == "failed"  # 本地 DNS 故障，不是策略拒绝
    failed = [record for record in sink.records if record.event_type == "tool_failed"]
    assert failed[0].payload["reason_code"] == "resolve_failed"
