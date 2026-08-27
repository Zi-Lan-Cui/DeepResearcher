import asyncio
import re

import pytest
from langchain_core.messages import AIMessage

from deepsearch_agent.agents.writer import CompleteReport, ReadEvidence, ReportWriter
from deepsearch_agent.config import AgentConfig, LLMRetryConfig
from deepsearch_agent.evidence import EvidenceExtractor
from deepsearch_agent.evidence.models import Evidence, EvidenceExtraction, ExtractedEvidence
from deepsearch_agent.evidence.retrieval import select_blocks
from deepsearch_agent.llm import LLMConfigurationError, structured
from deepsearch_agent.observability.events import JsonlSink
from deepsearch_agent.orchestration.nodes import (
    reflection,
    render_final_report_node,
)
from deepsearch_agent.reporting import (
    no_evidence_blockers,
    render_final_report,
    validate_and_bind,
)
from deepsearch_agent.reporting.validation import DraftProtocolError
from deepsearch_agent.schemas import (
    Citation,
    MarkdownReportDraft,
    ParagraphBinding,
    ReflectionDecision,
    ReportBrief,
    ReviewIssue,
    WriterDirective,
    WriterResult,
)


def _ev(
    evidence_id,
    claim,
    *,
    quote=None,
    url="https://example.com/a",
    support="direct",
    direction="",
    retrieval="origin_fetch",
):
    """构造合法 Evidence 模型;State 通道已类型化,测试 fixture 不再用裸 dict。"""
    return Evidence(
        evidence_id=evidence_id,
        subtask_id="r1-1",
        research_direction=direction,
        claim=claim,
        quote=quote if quote is not None else claim,
        source_url=url,
        support=support,
        retrieval_method=retrieval,
    )


def _binding(text, evidence_ids=None, kind=None):
    if kind is None:
        kind = "evidence" if evidence_ids else "transition"
    return ParagraphBinding(text=text, kind=kind, evidence_ids=evidence_ids or [])


def _cite(source_id, claim="事实", quote="原文", url=""):
    return Citation(id=source_id, claim=claim, quote=quote, url=url)


REPORT_BRIEF = {
    "answer_goal": "回答测试问题",
    "covered_topics": [
        {"topic": "核心结论", "role": "主线", "reason": "直接回答问题", "required": True}
    ],
    "required_points": ["给出有来源的结论"],
    "caveats": [],
}


class _WriterToolRunnable:
    def __init__(self, draft_factory):
        self.draft_factory = draft_factory

    async def ainvoke(self, messages):
        if not any(message.type == "tool" for message in messages):
            ids = re.findall(r"evidence_id=([^ |\\n]+)", str(messages[-1].content))
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": ReadEvidence.__name__,
                        "args": {"evidence_ids": ids, "reason": "测试读取目录中的证据"},
                        "id": "read-1",
                    }
                ],
            )
        draft = await self.draft_factory(object(), MarkdownReportDraft, messages)
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": CompleteReport.__name__,
                    "args": draft.model_dump(),
                    "id": "complete-1",
                }
            ],
        )


class _WriterToolLLM:
    def __init__(self, draft_factory):
        self.draft_factory = draft_factory

    def bind_tools(self, _tools, **_kwargs):
        return _WriterToolRunnable(self.draft_factory)


def _writer_llm(draft_factory):
    return _WriterToolLLM(draft_factory)


def _write_with(draft_output, state, *, event_sink=None):
    """用 Writer 工具协议运行测试；不绕过 ReadEvidence/CompleteReport。"""
    state = dict(state)
    state.setdefault(
        "writer_directive",
        WriterDirective(
            query=state.get("clarified_query", "测试问题"),
            report_brief=ReportBrief.model_validate(REPORT_BRIEF),
            research_status="completed",
            generation_mode="full",
        ),
    )
    return asyncio.run(
        ReportWriter(
            _writer_llm(draft_output),
            AgentConfig(),
            render_incomplete=lambda _state: "incomplete",
            event_sink=event_sink,
        ).run(state)
    )


def blocks():
    return [
        {
            "block_id": "b-0",
            "block_type": "heading",
            "text": "性能测试",
            "heading_path": ["性能测试"],
            "order": 0,
        },
        {
            "block_id": "b-1",
            "block_type": "paragraph",
            "text": "在相同硬件环境下，A 的平均延迟为 20ms。",
            "heading_path": ["性能测试"],
            "order": 1,
        },
        {
            "block_id": "b-2",
            "block_type": "paragraph",
            "text": "该结果仅适用于英文数据集。",
            "heading_path": ["性能测试"],
            "order": 2,
        },
        {
            "block_id": "b-3",
            "block_type": "paragraph",
            "text": "文章还讨论了部署成本。",
            "heading_path": ["成本"],
            "order": 3,
        },
    ]


def test_lexical_retrieval_keeps_adjacent_qualification():
    selected = select_blocks("A 平均延迟", blocks(), top_k=1, window=1)
    assert [block["block_id"] for block in selected] == ["b-0", "b-1", "b-2"]


def test_evidence_extractor_requires_llm_at_construction():
    with pytest.raises(LLMConfigurationError):
        EvidenceExtractor(None)


def test_llm_empty_evidence_does_not_fall_back_to_source_title(monkeypatch):
    async def empty_extraction(llm, schema, messages, **kwargs):
        return EvidenceExtraction(evidences=[])

    monkeypatch.setattr("deepsearch_agent.evidence.extractor.ainvoke_structured", empty_extraction)
    document = {
        "title": "只有标题的来源",
        "final_url": "https://example.com",
        "text": "这段内容没有支持当前研究任务的事实。",
        "blocks": blocks(),
    }
    result = {"title": "搜索标题", "url": "https://example.com", "snippet": "", "score": 0.8}

    evidences = asyncio.run(
        EvidenceExtractor(llm=object()).aextract(
            {
                "id": "r1-1",
                "question": "不存在的主题",
                "type": "search",
                "status": "pending",
                "assigned_agent": "search",
            },
            document,
            result,
        )
    )

    assert evidences == []


def test_long_document_is_extracted_from_structured_chunks_not_bm25_selection(monkeypatch):
    seen_contexts = []

    async def extract_each_chunk(llm, schema, messages, **kwargs):
        context = messages[-1].content
        seen_contexts.append(context)
        if "证据甲" in context:
            return EvidenceExtraction(
                evidences=[
                    ExtractedEvidence(
                        claim="事实甲", quote="证据甲", support="direct", confidence=0.9
                    )
                ]
            )
        if "证据乙" in context:
            return EvidenceExtraction(
                evidences=[
                    ExtractedEvidence(
                        claim="事实乙", quote="证据乙", support="direct", confidence=0.9
                    )
                ]
            )
        return EvidenceExtraction(evidences=[])

    monkeypatch.setattr(
        "deepsearch_agent.evidence.extractor.ainvoke_structured", extract_each_chunk
    )
    document = {
        "title": "长文",
        "final_url": "https://example.com/long",
        "text": "证据甲\n无关段落\n证据乙",
        "blocks": [
            {
                "block_id": "b-1",
                "block_type": "paragraph",
                "text": "证据甲",
                "heading_path": ["甲"],
                "order": 1,
            },
            {
                "block_id": "b-2",
                "block_type": "paragraph",
                "text": "无关段落",
                "heading_path": ["中"],
                "order": 2,
            },
            {
                "block_id": "b-3",
                "block_type": "paragraph",
                "text": "证据乙",
                "heading_path": ["乙"],
                "order": 3,
            },
        ],
    }
    result = {"title": "长文", "url": "https://example.com/long", "snippet": "", "score": 0.8}
    extractor = EvidenceExtractor(llm=object(), input_budget_tokens=5, chunk_concurrency=2)

    outcome = asyncio.run(
        extractor.aextract_result(
            {
                "id": "r1-1",
                "question": "不匹配的查询词",
                "type": "search",
                "status": "pending",
                "assigned_agent": "search",
            },
            document,
            result,
        )
    )

    assert outcome.strategy == "structured_chunks"
    assert outcome.chunk_count == 3
    assert {item.claim for item in outcome.evidences} == {"事实甲", "事实乙"}
    assert len(seen_contexts) == 3


def test_evidence_extractor_keeps_successful_chunks_when_one_chunk_fails(monkeypatch):
    calls = 0

    async def extract_one_chunk(llm, schema, messages, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("one chunk timed out")
        context = str(messages[-1].content)
        quote = next(
            value
            for value in ("甲段落", "乙段落", "丙段落", "甲", "乙", "丙", "段", "落")
            if value in context
        )
        return EvidenceExtraction(
            evidences=[ExtractedEvidence(claim="保留下来的事实", quote=quote)]
        )

    monkeypatch.setattr(
        "deepsearch_agent.evidence.extractor.ainvoke_structured", extract_one_chunk
    )
    document = {
        "title": "部分失败来源",
        "final_url": "https://example.com/partial",
        "text": "甲段落\n乙段落\n丙段落",
        "blocks": [
            {"block_id": "b-1", "block_type": "paragraph", "text": "甲段落", "heading_path": [], "order": 1},
            {"block_id": "b-2", "block_type": "paragraph", "text": "乙段落", "heading_path": [], "order": 2},
            {"block_id": "b-3", "block_type": "paragraph", "text": "丙段落", "heading_path": [], "order": 3},
        ],
    }
    outcome = asyncio.run(
        EvidenceExtractor(llm=object(), input_budget_tokens=3, chunk_concurrency=2).aextract_result(
            {"id": "r1-1", "question": "方向", "type": "search", "status": "pending", "assigned_agent": "search"},
            document,
            {"title": "部分失败来源", "url": document["final_url"], "snippet": "", "score": 0.8},
        )
    )
    assert outcome.failed_chunk_count == 1
    assert len(outcome.evidences) >= 1


def test_structured_output_uses_json_mode_and_supplies_json_contract(monkeypatch):
    captured = {}

    class FakeRunnable:
        async def ainvoke(self, messages):
            captured["messages"] = messages
            return EvidenceExtraction(evidences=[])

    class FakeLLM:
        def with_structured_output(self, schema, **kwargs):
            captured["schema"] = schema
            captured["kwargs"] = kwargs
            return FakeRunnable()

    monkeypatch.setattr(structured, "with_transport_retry", lambda runnable, policy: runnable)
    result = asyncio.run(
        structured.ainvoke_structured(
            structured.LLMInvoker(FakeLLM(), LLMRetryConfig()),
            EvidenceExtraction,
            [],
        )
    )

    assert result == EvidenceExtraction(evidences=[])
    assert captured["schema"] is EvidenceExtraction
    assert captured["kwargs"] == {"method": "json_mode"}
    assert "合法 JSON object" in captured["messages"][0].content
    assert '"evidences"' in captured["messages"][0].content


def test_structured_output_binds_optional_request_kwargs(monkeypatch):
    captured = {}

    class FakeRunnable:
        def bind(self, **kwargs):
            captured["request_kwargs"] = kwargs
            return self

        async def ainvoke(self, messages):
            return EvidenceExtraction(evidences=[])

    class FakeLLM:
        def with_structured_output(self, schema, **kwargs):
            return FakeRunnable()

    monkeypatch.setattr(structured, "with_transport_retry", lambda runnable, policy: runnable)
    asyncio.run(
        structured.ainvoke_structured(
            structured.LLMInvoker(FakeLLM(), LLMRetryConfig()),
            EvidenceExtraction,
            [],
            request_kwargs={"extra_body": {"thinking": {"type": "disabled"}}},
        )
    )

    assert captured["request_kwargs"] == {"extra_body": {"thinking": {"type": "disabled"}}}


def test_evidence_prompt_contains_json_example(monkeypatch):
    captured = {}

    async def fake_invoke(llm, schema, messages, **kwargs):
        captured["messages"] = messages
        return EvidenceExtraction(
            evidences=[
                ExtractedEvidence(
                    claim="A 的平均延迟为 20ms",
                    quote="在相同硬件环境下，A 的平均延迟为 20ms。",
                    support="direct",
                    confidence=0.8,
                )
            ]
        )

    monkeypatch.setattr("deepsearch_agent.evidence.extractor.ainvoke_structured", fake_invoke)
    document = {
        "title": "测试来源",
        "final_url": "https://example.com",
        "text": "\n".join(block["text"] for block in blocks()),
        "blocks": blocks(),
    }
    result = {"title": "延迟测试", "url": "https://example.com", "snippet": "", "score": 0.8}
    evidences = asyncio.run(
        EvidenceExtractor(llm=object()).aextract(
            {
                "id": "r1-1",
                "question": "A 平均延迟",
                "type": "search",
                "status": "pending",
                "assigned_agent": "search",
            },
            document,
            result,
        )
    )

    assert evidences
    prompt = captured["messages"][0].content
    assert "JSON 示例" in prompt
    assert '{"evidences"' in prompt


def test_evidence_extraction_records_raw_result_and_rejected_candidate(tmp_path, monkeypatch):
    async def fake_invoke(_llm, _schema, _messages, **kwargs):
        return EvidenceExtraction(
            evidences=[
                ExtractedEvidence(
                    claim="不存在于正文的结论",
                    quote="不存在于正文的引文",
                    support="direct",
                    confidence=0.8,
                )
            ]
        )

    monkeypatch.setattr("deepsearch_agent.evidence.extractor.ainvoke_structured", fake_invoke)
    sink_path = tmp_path / "events.jsonl"
    outcome = asyncio.run(
        EvidenceExtractor(llm=object(), event_sink=JsonlSink(sink_path)).aextract_result(
            {
                "id": "task-1",
                "question": "测试问题",
                "type": "search",
                "status": "pending",
                "assigned_agent": "research",
            },
            {
                "title": "测试来源",
                "final_url": "https://example.com/source",
                "text": "正文内容",
                "blocks": [
                    {
                        "block_id": "b-1",
                        "block_type": "paragraph",
                        "text": "正文内容",
                        "heading_path": [],
                        "order": 0,
                    }
                ],
            },
            {"url": "https://example.com/source", "title": "测试来源"},
        )
    )

    assert outcome.evidences == []
    assert outcome.validation_rejected_count == 1
    events = sink_path.read_text(encoding="utf-8")
    assert '"event_type": "evidence_llm_response"' in events
    assert '"candidate_count": 1' in events


def test_evidence_extractor_limits_model_output_to_configured_budget(monkeypatch):
    captured = {}

    async def fake_invoke(llm, schema, messages, **kwargs):
        captured["prompt"] = messages[0].content
        return EvidenceExtraction(
            evidences=[
                ExtractedEvidence(
                    claim="事实一",
                    quote="在相同硬件环境下，A 的平均延迟为 20ms。",
                    support="direct",
                    confidence=0.9,
                ),
                ExtractedEvidence(
                    claim="事实二",
                    quote="该结果仅适用于英文数据集。",
                    support="direct",
                    confidence=0.8,
                ),
            ]
        )

    monkeypatch.setattr("deepsearch_agent.evidence.extractor.ainvoke_structured", fake_invoke)
    document = {
        "title": "测试来源",
        "final_url": "https://example.com",
        "text": "\n".join(block["text"] for block in blocks()),
        "blocks": blocks(),
    }
    result = {"title": "延迟测试", "url": "https://example.com", "snippet": "", "score": 0.8}

    evidences = asyncio.run(
        EvidenceExtractor(llm=object(), max_evidences=1).aextract(
            {
                "id": "r1-1",
                "question": "A 平均延迟",
                "type": "search",
                "status": "pending",
                "assigned_agent": "search",
            },
            document,
            result,
        )
    )

    assert len(evidences) == 1
    assert "最多返回 1 条 Evidence" in captured["prompt"]


def test_writer_requires_llm_at_construction():
    with pytest.raises(LLMConfigurationError):
        ReportWriter(
            None,
            AgentConfig(),
            render_incomplete=object(),
        )


def test_writer_filters_evidence_by_configured_minimum_support():
    state = {
        "clarified_query": "测试问题",
        "report_brief": REPORT_BRIEF,
        "writer_directive": WriterDirective(
            query="测试问题",
            report_brief=ReportBrief.model_validate(REPORT_BRIEF),
            research_status="completed",
            generation_mode="full",
        ),
        "evidences": [
            _ev(
                "e1",
                "搜索摘要中的事实",
                quote="搜索摘要中的事实。",
                url="https://example.com",
                support="partial",
                retrieval="search_summary",
            )
        ],
    }

    async def draft_output(*_args, **_kwargs):
        return MarkdownReportDraft(
            selected_evidence_ids=["e1"],
            markdown="## 结论\n\n搜索摘要中的事实。[[cite:e1]]",
        )

    strict_writer = ReportWriter(
        _writer_llm(draft_output),
        AgentConfig(writer_minimum_support="direct"),
        render_incomplete=lambda _state: "incomplete",
    )
    strict_result = asyncio.run(strict_writer.run(state))
    assert strict_result["answer_mode"] == "research_incomplete"
    assert "最低支持等级 `direct`" in strict_result["report"]

    partial_writer = ReportWriter(
        _writer_llm(draft_output),
        AgentConfig(writer_minimum_support="partial"),
        render_incomplete=lambda _state: "incomplete",
    )
    partial_result = asyncio.run(partial_writer.run(state))
    assert partial_result["writer"].status == "completed"
    assert "[[cite:e1]]" in partial_result["report_draft"]


def test_writer_result_serializes_nested_citation_models_for_graph_state():
    result = WriterResult(
        report_draft="## 结论\n\n事实。[[cite:e1]]",
        answer_mode="deep_research",
        citations=[Citation(id="e1", url="https://example.com/a", quote="原文", claim="事实")],
        paragraph_bindings=[ParagraphBinding(text="事实。", kind="evidence", evidence_ids=["e1"])],
    ).state_update()

    assert result["citations"] == [_cite("e1", url="https://example.com/a").model_dump()]
    assert result["paragraph_bindings"] == [_binding("事实。", ["e1"]).model_dump()]
    assert result["report_draft"] == "## 结论\n\n事实。[[cite:e1]]"


def test_writer_binds_markdown_cite_to_explicit_evidence():
    async def draft_output(llm, schema, messages, **kwargs):
        assert schema is MarkdownReportDraft
        return MarkdownReportDraft(
            selected_evidence_ids=["e1"], markdown="## 结论\n\nA 的平均延迟为 20ms。[[cite:e1]]"
        )

    result = _write_with(
        draft_output,
        {
            "clarified_query": "A 的性能",
            "report_brief": REPORT_BRIEF,
            "evidences": [
                _ev(
                    "e1",
                    "A 的平均延迟为 20ms",
                    quote="A 的平均延迟为 20ms。",
                    url="https://example.com/a",
                )
            ],
        },
    )

    # Writer 产出 evidence_id 键的草稿与绑定，不渲染编号
    assert result["paragraph_bindings"] == [_binding("A 的平均延迟为 20ms。", ["e1"]).model_dump()]
    assert "[[cite:e1]]" in result["report_draft"]
    assert "[来源1]" not in result["report_draft"]
    assert result["citations"] == [
        _cite(
            "e1",
            claim="A 的平均延迟为 20ms",
            quote="A 的平均延迟为 20ms。",
            url="https://example.com/a",
        ).model_dump()
    ]


def test_render_final_report_renumbers_by_first_appearance_and_keeps_quotes():
    report = render_final_report(
        clarified_query="A 的性能",
        current_round=2,
        evidence_count=2,
        body=(
            "## 结论\n\n"
            "第二条 Evidence 支持的结论。[[cite:e2]]\n\n"
            "第一条 Evidence 补充该结论。[[cite:e1]]"
        ),
        citations=[
            Citation(
                id="e1",
                url="https://example.com/one",
                title="来源一",
                quote="第一条。",
                claim="第一条",
            ),
            Citation(
                id="e2",
                url="https://example.com/two",
                title="来源二",
                quote="第二条。",
                claim="第二条",
            ),
        ],
    )

    # 编号按正文首现顺序：e2 → 来源1，e1 → 来源2
    assert "[来源1]" in report
    assert "[来源2]" in report
    assert report.index("[来源1]") < report.index("[来源2]")
    assert "## 参考来源" in report
    # 参考来源表保留可审计 quote，最终交付物不丢 chunk
    assert "「第二条。」" in report
    assert "「第一条。」" in report
    assert "https://example.com/two" in report
    assert "https://example.com/one" in report


def test_render_final_report_ignores_cite_markers_inside_code():
    report = render_final_report(
        clarified_query="测试代码隔离",
        current_round=1,
        evidence_count=1,
        body=(
            "## 示例\n\n"
            "`[[cite:e2]]`\n\n"
            "```text\n[[cite:e2]]\n```\n\n"
            "真正需要引用的结论。[[cite:e1]]"
        ),
        citations=[
            Citation(
                id="e1",
                url="https://example.com/one",
                title="来源一",
                quote="第一条。",
                claim="第一条",
            ),
            Citation(
                id="e2",
                url="https://example.com/two",
                title="来源二",
                quote="第二条。",
                claim="第二条",
            ),
        ],
    )

    assert "[[cite:e2]]" in report  # 代码块内原样保留
    assert "[来源1]" in report
    assert "https://example.com/two" not in report  # 代码里的伪标记不进入参考表


def test_render_final_report_node_routes_failure_paths():
    # 写作失败 → 兜底渲染
    result = asyncio.run(
        render_final_report_node(
            {
                "clarified_query": "问题",
                "writer": {"status": "exhausted", "feedback": "引用协议失败"},
            },
            None,
        )
    )
    assert result["answer_mode"] == "research_incomplete"
    assert "报告写作未能完成" in result["report"]

    # 审阅拒绝且恢复耗尽 → 兜底渲染
    result = asyncio.run(
        render_final_report_node(
            {
                "clarified_query": "问题",
                "review": {"status": "rejected", "feedback": "核心结论缺少来源"},
            },
            None,
        )
    )
    assert result["answer_mode"] == "research_incomplete"
    assert "核心结论缺少来源" in result["report"]

    # 快乐路径:草稿 + citations → 渲染
    result = asyncio.run(
        render_final_report_node(
            {
                "clarified_query": "A 的性能",
                "current_round": 1,
                "evidence_count": 1,
                "report_draft": "## 结论\n\nA 的平均延迟为 20ms。[[cite:e1]]",
                "citations": [
                    Citation(
                        id="e1",
                        url="https://example.com/a",
                        title="来源",
                        quote="原文。",
                        claim="事实",
                    )
                ],
            },
            None,
        )
    )
    assert "[来源1]" in result["report"]
    assert "## 参考来源" in result["report"]
    assert "「原文。」" in result["report"]


def test_writer_audit_events_keep_generated_draft(tmp_path):
    async def draft_output(*_args, **_kwargs):
        return MarkdownReportDraft(
            selected_evidence_ids=["e1"],
            markdown="## 结论\n\nA 的平均延迟为 20ms。[[cite:e1]]",
        )

    sink_path = tmp_path / "events.jsonl"
    agent = ReportWriter(
        _writer_llm(draft_output),
        AgentConfig(),
        render_incomplete=lambda _state: "incomplete",
        event_sink=JsonlSink(sink_path),
    )
    asyncio.run(
        agent.run(
            {
                "clarified_query": "A 的性能",
                "report_brief": REPORT_BRIEF,
                "writer_directive": WriterDirective(
                    query="A 的性能",
                    report_brief=ReportBrief.model_validate(REPORT_BRIEF),
                    research_status="completed",
                    generation_mode="full",
                ),
                "evidences": [
                    _ev(
                        "e1",
                        "A 的平均延迟为 20ms",
                        quote="A 的平均延迟为 20ms。",
                        url="https://example.com/a",
                    )
                ],
            }
        )
    )

    events = sink_path.read_text(encoding="utf-8")
    assert '"event_type": "writer_evidence_read"' in events
    assert '"event_type": "writer_draft_validated"' in events
    assert '"event_type": "writer_draft_ready"' in events


def test_writer_uses_direction_grouped_evidence_index_without_raw_quote():
    captured = {}

    async def draft_output(llm, schema, messages, **kwargs):
        if "prompt" not in captured:
            captured["prompt"] = next(
                message.content
                for message in messages
                if "可选 Evidence 目录" in str(message.content)
            )
        return MarkdownReportDraft(
            selected_evidence_ids=["r1-1-ev-1"],
            markdown="## 结论\n\nA 的平均延迟为 20ms。[[cite:r1-1-ev-1]]",
        )

    result = _write_with(
        draft_output,
        {
            "clarified_query": "A 的性能",
            "report_brief": REPORT_BRIEF,
            "evidences": [
                _ev(
                    "r1-1-ev-1",
                    "A 的平均延迟为 20ms",
                    quote="这是不应进入 Writer 上下文的完整原文引文。",
                    url="https://example.com/a",
                    direction="A 的性能与测试条件",
                )
            ],
        },
    )

    assert "[研究方向] A 的性能与测试条件" in captured["prompt"]
    assert "Supervisor 的报告任务书" in captured["prompt"]
    assert "回答测试问题" in captured["prompt"]
    assert "claim：A 的平均延迟为 20ms" in captured["prompt"]
    assert "这是不应进入 Writer 上下文的完整原文引文。" not in captured["prompt"]
    assert result["writer"].status == "completed"
    assert result["writer"].selected_evidence_ids == ["r1-1-ev-1"]


def test_writer_repairs_selection_from_valid_cites_without_regeneration():
    async def draft_output(llm, schema, messages, **kwargs):
        return MarkdownReportDraft(
            selected_evidence_ids=["e1"],
            markdown="## 结论\n\n未选择的 Evidence。[[cite:e2]]",
        )

    result = _write_with(
        draft_output,
        {
            "clarified_query": "测试选择集合",
            "report_brief": REPORT_BRIEF,
            "evidences": [
                _ev("e1", "事实一", quote="原文一", url="https://example.com/one"),
                _ev("e2", "事实二", quote="原文二", url="https://example.com/two"),
            ],
        },
    )

    assert result["writer"].status == "completed"
    assert result["writer"].selected_evidence_ids == ["e1", "e2"]


def test_writer_preserves_uncited_conclusion_for_reflection():
    async def draft_output(llm, schema, messages, **kwargs):
        return MarkdownReportDraft(
            selected_evidence_ids=["e1"],
            markdown=(
                "## 分析\n\n"
                "A 的平均延迟为 20ms。[[cite:e1]]\n\n"
                "因此，这项结果应结合测试条件理解，不能单独外推到所有场景。"
            ),
        )

    result = _write_with(
        draft_output,
        {
            "clarified_query": "A 的性能",
            "report_brief": REPORT_BRIEF,
            "evidences": [
                _ev(
                    "e1",
                    "A 的平均延迟为 20ms",
                    quote="A 的平均延迟为 20ms。",
                    url="https://example.com/a",
                )
            ],
        },
    )

    assert result["paragraph_bindings"] == [
        _binding("A 的平均延迟为 20ms。", ["e1"]).model_dump(),
        _binding("因此，这项结果应结合测试条件理解，不能单独外推到所有场景。").model_dump(),
    ]


def test_writer_surfaces_last_cite_validation_diagnostic():
    async def draft_output(llm, schema, messages, **kwargs):
        return MarkdownReportDraft(
            selected_evidence_ids=["不存在的来源"],
            markdown="## 结论\n\n无效引用。[[cite:不存在的来源]]",
        )

    result = _write_with(
        draft_output,
        {
            "clarified_query": "A 的性能",
            "report_brief": REPORT_BRIEF,
            "evidences": [
                _ev(
                    "e1",
                    "A 的平均延迟为 20ms",
                    quote="A 的平均延迟为 20ms。",
                    url="https://example.com/a",
                )
            ],
        },
    )

    assert result["writer"].status == "exhausted"
    assert result["writer"].failure_kind == "citation_protocol"
    assert "不存在的来源" in result["writer"].feedback
    assert "[[cite:不存在的来源]]" in result["writer_draft"]


def test_writer_does_not_decide_to_restart_research():
    async def draft_output(llm, schema, messages, **kwargs):
        return MarkdownReportDraft(
            selected_evidence_ids=["e1"],
            markdown="## 结论\n\nA 是一种类型。[[cite:e1]]",
        )

    result = _write_with(
        draft_output,
        {
            "clarified_query": "A 与 B 有何差异",
            "report_brief": REPORT_BRIEF,
            "evidences": [
                _ev("e1", "A 是一种类型", quote="A 是一种类型。", url="https://example.com/a")
            ],
        },
    )

    assert result["writer"].status == "completed"
    assert result["writer"].selected_evidence_ids == ["e1"]


def test_writer_accepts_chinese_source_separators():
    async def draft_output(llm, schema, messages, **kwargs):
        return MarkdownReportDraft(
            selected_evidence_ids=["e2", "e1"],
            markdown="## 结论\n\n两个来源共同支撑的结论。[[cite:e2，e1]]",
        )

    result = _write_with(
        draft_output,
        {
            "clarified_query": "测试问题",
            "report_brief": REPORT_BRIEF,
            "evidences": [
                _ev("e1", "第一条事实", quote="第一条证据", url="https://one.test"),
                _ev("e2", "第二条事实", quote="第二条证据", url="https://two.test"),
            ],
        },
    )

    assert result["paragraph_bindings"] == [
        _binding("两个来源共同支撑的结论。", ["e2", "e1"], kind="synthesis").model_dump()
    ]


def test_writer_retries_invalid_evidence_binding_instead_of_falling_back():
    calls = 0

    async def draft_output(llm, schema, messages, **kwargs):
        nonlocal calls
        calls += 1
        source_id = "不存在的来源" if calls == 1 else "e1"
        return MarkdownReportDraft(
            selected_evidence_ids=[source_id],
            markdown=(
                "## 结论\n\n"
                f"A 的平均延迟为 20ms。[[cite:{source_id}]]\n\n"
                "这一结果仅适用于给定测试条件。"
            ),
        )

    result = _write_with(
        draft_output,
        {
            "clarified_query": "A 的性能",
            "report_brief": REPORT_BRIEF,
            "evidences": [
                _ev(
                    "e1",
                    "A 的平均延迟为 20ms",
                    quote="A 的平均延迟为 20ms。",
                    url="https://example.com/a",
                )
            ],
        },
    )

    assert calls == 2
    assert result["paragraph_bindings"][0]["evidence_ids"] == ["e1"]
    assert "这一结果仅适用于给定测试条件。" in result["report_draft"]


def test_writer_rejects_manual_reference_section_and_numbering():
    async def draft_output(llm, schema, messages, **kwargs):
        return MarkdownReportDraft(
            selected_evidence_ids=["e1"],
            markdown="## 结论\n\nA 的平均延迟为 20ms。 [来源1]\n\n## 参考来源\n- [来源1] 示例: https://example.com/a",
        )

    result = _write_with(
        draft_output,
        {
            "clarified_query": "A 的性能",
            "report_brief": REPORT_BRIEF,
            "evidences": [
                _ev(
                    "e1",
                    "A 的平均延迟为 20ms",
                    quote="A 的平均延迟为 20ms。",
                    url="https://example.com/a",
                )
            ],
        },
    )

    assert result["writer"].status == "exhausted"
    assert result["writer"].failure_kind == "citation_protocol"


def test_writer_does_not_parse_cite_markers_inside_fenced_or_inline_code():
    async def draft_output(llm, schema, messages, **kwargs):
        return MarkdownReportDraft(
            selected_evidence_ids=["e1"],
            markdown=(
                "## 示例\n\n"
                "`[[cite:e2]]`\n\n"
                "```text\n[[cite:e2]]\n```\n\n"
                "真正需要引用的结论。[[cite:e1]]"
            ),
        )

    result = _write_with(
        draft_output,
        {
            "clarified_query": "测试代码隔离",
            "report_brief": REPORT_BRIEF,
            "evidences": [
                _ev("e1", "第一条", quote="第一条。", url="https://example.com/one"),
                _ev("e2", "第二条", quote="第二条。", url="https://example.com/two"),
            ],
        },
    )

    assert result["paragraph_bindings"] == [
        _binding("真正需要引用的结论。", ["e1"]).model_dump()
    ]


def test_validate_and_bind_rejects_unknown_evidence_and_too_many_sources():
    with pytest.raises(DraftProtocolError, match="不存在的 Evidence"):
        validate_and_bind("事实。[[cite:unknown]]", {"e1": _ev("e1", "事实")})
    with pytest.raises(DraftProtocolError, match="最多绑定三个"):
        validate_and_bind(
            "事实。[[cite:e1,e2,e3,e4]]", {f"e{i}": _ev(f"e{i}", f"事实{i}") for i in range(1, 5)}
        )


def test_reflection_rejects_with_structured_evidence_feedback(monkeypatch):
    captured = {}

    async def review_output(llm, schema, messages):
        assert schema is ReflectionDecision
        captured["context"] = messages[-1].content
        return ReflectionDecision(
            feedback="需要能直接支持文学性评价的评论来源。",
            gaps=["缺少具体作品的文学性评价依据"],
            issues=[ReviewIssue(severity="fatal", reason="核心文学性结论缺少直接来源。")],
        )

    monkeypatch.setattr("deepsearch_agent.orchestration.nodes.ainvoke_structured", review_output)
    result = asyncio.run(
        reflection(
            {
                "clarified_query": "哪些作品文学性高",
                "report_brief": REPORT_BRIEF,
                "review_attempts": 0,
                "paragraph_bindings": [_binding("A 文学性高", ["e1"], kind="evidence")],
                "citations": [_cite("e1", claim="A 有复杂叙事", quote="A 有复杂叙事。")],
            },
            object(),
        )
    )

    assert result["review"].status == "rejected"
    assert result["review"].gaps == ["缺少具体作品的文学性评价依据"]
    assert "Supervisor 报告任务书" in captured["context"]


def test_reflection_requests_rewrite_when_evidence_is_sufficient(monkeypatch):
    async def review_output(llm, schema, messages):
        assert schema is ReflectionDecision
        return ReflectionDecision(
            feedback="将‘证明文学性’收窄为‘可作为获得认可的线索’。",
            gaps=[],
            issues=[ReviewIssue(severity="fatal", reason="核心结论把提名误写为文学性证明。")],
        )

    monkeypatch.setattr("deepsearch_agent.orchestration.nodes.ainvoke_structured", review_output)
    result = asyncio.run(
        reflection(
            {
                "clarified_query": "哪些作品文学性高",
                "review_attempts": 0,
                "paragraph_bindings": [_binding("A 的提名证明文学性", ["e1"], kind="evidence")],
                "citations": [_cite("e1", claim="A 获得提名", quote="A 获得提名。")],
            },
            object(),
        )
    )

    assert result["review"].status == "rejected"
    assert result["review"].gaps == []


def test_reflection_allows_warnings_without_rejecting_report(monkeypatch):
    async def review_output(llm, schema, messages):
        return ReflectionDecision(
            feedback="核心结论可交付；可选地收窄一处措辞。",
            issues=[
                ReviewIssue(
                    severity="warning",
                    claim="A 可能增强沉浸感。",
                    reason="来源只直接描述了 VR 体验。",
                    suggested_revision="保留‘可能’并标为分析。",
                )
            ],
        )

    monkeypatch.setattr("deepsearch_agent.orchestration.nodes.ainvoke_structured", review_output)
    result = asyncio.run(
        reflection(
            {
                "clarified_query": "A 有何体验特点",
                "paragraph_bindings": [_binding("A 可能增强沉浸感。", ["e1"], kind="synthesis")],
                "citations": [_cite("e1", claim="A 支持 VR 漫游", quote="A 支持 VR 漫游。")],
            },
            object(),
        )
    )

    assert result["review"].status == "approved"
    assert result["review"].issues[0].severity == "warning"


def test_no_evidence_blockers_preserves_failure_and_skip_diagnostics():
    blockers = no_evidence_blockers(
        {
            "task_results": [
                {
                    "failures": [
                        "https://example.com/a: HTTP 403",
                        "https://example.com/b: evidence extraction timed out",
                    ],
                    "skip_reasons": ["access_challenge", "evidence_empty"],
                },
                {
                    "failures": ["https://example.com/a: HTTP 403"],
                    "skip_reasons": ["access_challenge"],
                },
            ]
        }
    )

    assert "3 次候选来源读取或证据抽取失败" in blockers[0]
    assert "HTTP 403" in blockers[0]
    assert "access_challenge × 2" in blockers[1]


def test_rendered_report_preserves_auditable_quotes_in_reference_list():
    """渲染不变量：编号顺序 == 正文首现顺序，参考表逐行对应 quote，chunk 不丢。"""
    citations = [
        Citation(
            id="e1", url="https://one.test", title="来源一", quote="第一条原文。", claim="第一条"
        ),
        Citation(
            id="e2", url="https://two.test", title="来源二", quote="第二条原文。", claim="第二条"
        ),
    ]
    report = render_final_report(
        clarified_query="问题",
        current_round=1,
        evidence_count=2,
        body="第一条结论。[[cite:e1]]\n\n第二条结论。[[cite:e2]]",
        citations=citations,
    )

    assert report.index("[来源1]") < report.index("[来源2]")
    assert "「第一条原文。」" in report
    assert "「第二条原文。」" in report
    assert "来源一: https://one.test" in report
    assert "来源二: https://two.test" in report
