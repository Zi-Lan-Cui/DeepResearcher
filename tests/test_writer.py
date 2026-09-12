import asyncio

import pytest

from deepsearch_agent.agents.writer import ReportWriter
from deepsearch_agent.config import AgentConfig
from deepsearch_agent.llm import LLMConfigurationError
from deepsearch_agent.observability.events import JsonlSink
from deepsearch_agent.schemas import (
    Citation,
    MarkdownReportDraft,
    ParagraphBinding,
    ReportBrief,
    WriterDirective,
    WriterResult,
)
from fakes import REPORT_BRIEF, _WriterNoToolLLM, write_with, writer_llm
from fakes import make_binding as _binding
from fakes import make_citation as _cite
from fakes import make_evidence as _ev


def test_writer_requires_llm_at_construction():
    with pytest.raises(LLMConfigurationError):
        ReportWriter(
            None,
            AgentConfig(),
            render_incomplete=object(),
        )


def test_writer_never_receives_evidence_audit_chunk() -> None:
    evidence = _ev(
        "e1",
        "可公开的结论",
        quote="可公开的原文。",
        url="https://example.com/source",
    )
    evidence.audit_chunk = "SECRET_AUDIT_CONTEXT"
    observed_messages = []

    async def draft_output(_llm, _schema, messages, **_kwargs):
        observed_messages.extend(messages)
        return MarkdownReportDraft(
            selected_evidence_ids=["e1"],
            markdown="## 结论\n\n可公开的结论。[[cite:e1]]",
        )

    write_with(
        draft_output,
        {"clarified_query": "测试", "evidences": [evidence]},
    )

    assert "SECRET_AUDIT_CONTEXT" not in "\n".join(
        str(message.content) for message in observed_messages
    )


def test_writer_exposes_published_at_as_metadata_but_not_locator() -> None:
    evidence = _ev(
        "e1",
        "可公开的结论",
        quote="可公开的原文。",
        url="https://example.com/source",
    )
    evidence.published_at = "2026-07-04"
    evidence.locator.block_ids = ["private-block-id"]
    evidence.locator.heading_path = ["内部章节"]

    catalogue = ReportWriter._evidence_catalogue({"e1": evidence})

    assert "published_at=2026-07-04(搜索元信息)" in catalogue
    assert "private-block-id" not in catalogue
    assert "内部章节" not in catalogue


def test_writer_catalogue_preserves_topic_assignments_and_deduplicates_cards() -> None:
    first = _ev("e1", "共用事实", url="https://example.com/1")
    second = _ev("e2", "补充事实", url="https://example.com/2")
    brief = ReportBrief.model_validate(
        {
            "answer_goal": "回答问题",
            "covered_topics": [
                {
                    "topic": "主题 A",
                    "role": "定义",
                    "reason": "解释概念",
                    "evidence_ids": ["e1"],
                },
                {
                    "topic": "主题 B",
                    "role": "对比",
                    "reason": "说明差异",
                    "evidence_ids": ["e1", "e2"],
                },
            ],
        }
    )

    catalogue = ReportWriter._evidence_catalogue({"e1": first, "e2": second}, brief)

    assert "[写作主题] 主题 A" in catalogue
    assert "[写作主题] 主题 B" in catalogue
    assert "建议 Evidence：e1, e2" in catalogue
    assert catalogue.count("- evidence_id=e1 |") == 1
    assert catalogue.count("- evidence_id=e2 |") == 1


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
        writer_llm(draft_output),
        AgentConfig(writer_minimum_support="direct"),
        render_incomplete=lambda _state: "incomplete",
    )
    strict_result = asyncio.run(strict_writer.run(state))
    assert strict_result["answer_mode"] == "research_incomplete"
    assert "最低支持等级 `direct`" in strict_result["report"]

    partial_writer = ReportWriter(
        writer_llm(draft_output),
        AgentConfig(writer_minimum_support="partial"),
        render_incomplete=lambda _state: "incomplete",
    )
    partial_result = asyncio.run(partial_writer.run(state))
    assert partial_result["writer"].status == "completed"
    assert "[[cite:e1]]" in partial_result["report_draft"]


def test_writer_ends_without_tool_call_as_exhausted_submission():
    result = asyncio.run(
        ReportWriter(
            _WriterNoToolLLM(),
            AgentConfig(),
            render_incomplete=lambda _state: "incomplete",
        ).run(
            {
                "clarified_query": "测试问题",
                "writer_directive": WriterDirective(
                    query="测试问题",
                    report_brief=ReportBrief.model_validate(REPORT_BRIEF),
                    research_status="completed",
                    generation_mode="full",
                ),
                "evidences": [_ev("e1", "可验证事实")],
            }
        )
    )

    assert result["writer"].status == "exhausted"
    assert result["writer"].feedback == "Writer 未提交有效报告。"


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

    result = write_with(
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


def test_writer_audit_events_keep_generated_draft(tmp_path):
    async def draft_output(*_args, **_kwargs):
        return MarkdownReportDraft(
            selected_evidence_ids=["e1"],
            markdown="## 结论\n\nA 的平均延迟为 20ms。[[cite:e1]]",
        )

    sink_path = tmp_path / "events.jsonl"
    agent = ReportWriter(
        writer_llm(draft_output),
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


def test_writer_uses_deduplicated_evidence_index_without_raw_quote():
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

    result = write_with(
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

    assert "[Evidence 去重索引]" in captured["prompt"]
    assert "[研究方向]" not in captured["prompt"]
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

    result = write_with(
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

    result = write_with(
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

    result = write_with(
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

    result = write_with(
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

    result = write_with(
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

    result = write_with(
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

    result = write_with(
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

    result = write_with(
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

    assert result["paragraph_bindings"] == [_binding("真正需要引用的结论。", ["e1"]).model_dump()]


def test_writer_turn_logging_records_tool_calls_and_stop_reason(tmp_path):
    """Writer 耗尽轮次时，逐轮记录模型调用的工具与终止原因（观测性回归）。"""
    import json

    from langchain_core.messages import AIMessage

    from deepsearch_agent.evidence.models import Evidence
    from deepsearch_agent.observability.events import JsonlSink
    from deepsearch_agent.schemas import CoveredTopic, ReportBrief, WriterDirective

    class LoopingLLM:
        """只调用 ReadEvidence、永不提交 CompleteReport 的假模型。"""

        def bind_tools(self, _tools, **_kwargs):
            return self

        async def ainvoke(self, _messages):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "ReadEvidence",
                        "args": {"evidence_ids": ["e1"], "reason": "写作需要"},
                        "id": "c1",
                    }
                ],
            )

    config = AgentConfig(writer_max_turns=3)
    sink = JsonlSink(tmp_path / "events.jsonl")
    ev = Evidence(
        evidence_id="e1",
        subtask_id="t1",
        research_direction="测试方向",
        claim="孟子主张性善",
        quote="可验证原文。",
        source_url="https://example.com/1",
    )
    state = {
        "query": "测试",
        "evidences": [ev],
        "writer_directive": WriterDirective(
            query="测试",
            report_brief=ReportBrief(
                answer_goal="回答",
                covered_topics=[CoveredTopic(topic="性善", role="核心", reason="直接回答")],
            ),
            research_status="completed",
            generation_mode="full",
        ),
    }
    writer = ReportWriter(
        LoopingLLM(), config, render_incomplete=lambda _s: "incomplete", event_sink=sink
    )
    result = asyncio.run(writer.run(state))

    assert result["writer"].status == "exhausted"
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    turns = [item for item in events if item["event_type"] == "writer_model_turn"]
    finished = [item for item in events if item["event_type"] == "writer_agent_finished"]

    assert len(turns) == config.writer_max_turns
    assert all(item["payload"]["tool_names"] == ["ReadEvidence"] for item in turns)
    assert [item["payload"]["turn"] for item in turns] == [1, 2, 3]
    assert finished[0]["payload"]["stop_reason"] == "model_call_limit_exceeded"
    assert finished[0]["payload"]["turns"] == config.writer_max_turns


def _writer_inline_test_setup(tmp_path, llm):
    """构造 Writer 内联草稿测试共用的 Evidence、指令与事件 sink。"""
    from deepsearch_agent.evidence.models import Evidence
    from deepsearch_agent.observability.events import JsonlSink
    from deepsearch_agent.schemas import CoveredTopic, ReportBrief, WriterDirective

    sink = JsonlSink(tmp_path / "events.jsonl")
    ev = Evidence(
        evidence_id="e1",
        subtask_id="t1",
        research_direction="测试方向",
        claim="孟子主张性善",
        quote="可验证原文。",
        source_url="https://example.com/1",
    )
    state = {
        "query": "测试",
        "evidences": [ev],
        "writer_directive": WriterDirective(
            query="测试",
            report_brief=ReportBrief(
                answer_goal="回答",
                covered_topics=[CoveredTopic(topic="性善", role="核心", reason="直接回答")],
            ),
            research_status="completed",
            generation_mode="full",
        ),
    }
    writer = ReportWriter(
        llm, AgentConfig(), render_incomplete=lambda _s: "incomplete", event_sink=sink
    )
    return writer, state, sink


class _InlineReportLLM:
    """先读 Evidence，再把整篇报告当收尾正文直接输出、跳过 CompleteReport。"""

    def __init__(self, cite_id: str):
        self.cite_id = cite_id
        self.turn = 0

    def bind_tools(self, _tools, **_kwargs):
        return self

    async def ainvoke(self, _messages):
        from langchain_core.messages import AIMessage

        self.turn += 1
        if self.turn == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "ReadEvidence",
                        "args": {"evidence_ids": ["e1"], "reason": "写作需要"},
                        "id": "c1",
                    }
                ],
            )
        paragraph = f"孟子认为人性本善，四端说是其核心论证。[[cite:{self.cite_id}]]"
        return AIMessage(content=("## 一、先秦\n\n" + paragraph * 12))


def test_writer_recovers_inline_draft_without_complete_report(tmp_path):
    """模型跳过 CompleteReport 直接输出合规正文时，本地校验通过并按已提交草稿进入审阅。"""
    writer, state, _ = _writer_inline_test_setup(tmp_path, _InlineReportLLM("e1"))
    result = asyncio.run(writer.run(state))

    assert result["writer"].status == "completed"
    assert result["run"].phase == "reviewing"
    assert "[[cite:e1]]" in result["report_draft"]
    assert result["writer"].selected_evidence_ids == ["e1"]

    import json

    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(item["event_type"] == "writer_inline_draft_recovered" for item in events)


def test_writer_rejects_invalid_inline_draft_but_keeps_text(tmp_path):
    """内联正文引用了未读取的 Evidence：按失败处理，但正文不再被整篇丢弃。"""
    writer, state, _ = _writer_inline_test_setup(tmp_path, _InlineReportLLM("e9"))
    result = asyncio.run(writer.run(state))

    assert result["writer"].status == "exhausted"
    assert len(result["writer_draft"]) > 300
    assert "引用校验" in result["writer"].feedback or "不存在" in result["writer"].feedback


class _NudgeAwareLLM:
    """先违规输出正文；收到守卫踢回提示后改用 CompleteReport 正规提交。"""

    def __init__(self):
        self.turn = 0

    def bind_tools(self, _tools, **_kwargs):
        return self

    async def ainvoke(self, messages):
        from langchain_core.messages import HumanMessage

        self.turn += 1
        if self.turn == 1:
            from langchain_core.messages import AIMessage

            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "ReadEvidence",
                        "args": {"evidence_ids": ["e1"], "reason": "写作需要"},
                        "id": "c1",
                    }
                ],
            )
        report = "## 一、先秦\n\n" + "孟子认为人性本善，四端说是其核心论证。[[cite:e1]]" * 12
        nudged = any(
            isinstance(message, HumanMessage) and "不算提交" in str(message.content)
            for message in messages
        )
        from langchain_core.messages import AIMessage

        if not nudged:
            return AIMessage(content=report)
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "CompleteReport",
                    "args": {"selected_evidence_ids": ["e1"], "markdown": report},
                    "id": "cr1",
                }
            ],
        )


def _event_types(tmp_path):
    import json

    return [
        json.loads(line)["event_type"]
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def test_writer_guard_nudges_text_only_output_back_into_tool_submission(tmp_path):
    """提交前输出纯文本被守卫踢回，模型改用 CompleteReport 时按正规路径完成。"""
    writer, state, _ = _writer_inline_test_setup(tmp_path, _NudgeAwareLLM())
    result = asyncio.run(writer.run(state))

    assert result["writer"].status == "completed"
    assert "[[cite:e1]]" in result["report_draft"]
    events = _event_types(tmp_path)
    assert events.count("writer_tool_loop_nudged") == 1
    # 走的是工具提交路径，而非内联救回
    assert "writer_draft_ready" in events
    assert "writer_inline_draft_recovered" not in events


def test_writer_guard_bounded_then_recovery_catches(tmp_path):
    """模型坚持输出正文：守卫最多踢回 2 次后放行，由内联救回兜底，不死循环。"""
    writer, state, _ = _writer_inline_test_setup(tmp_path, _InlineReportLLM("e1"))
    result = asyncio.run(writer.run(state))

    assert result["writer"].status == "completed"
    events = _event_types(tmp_path)
    assert events.count("writer_tool_loop_nudged") == 2
    assert "writer_inline_draft_recovered" in events


class _BulkReadThenWriteLLM:
    """模拟模型的合理用法：一次请求多于批量上限的 id，靠显式截断信号补齐。"""

    def bind_tools(self, _tools, **_kwargs):
        return self

    async def ainvoke(self, messages, **_kwargs):
        from langchain_core.messages import AIMessage

        history = "\n".join(str(message.content) for message in messages)
        read_calls = history.count('"read_count"')
        if read_calls == 0:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "ReadEvidence",
                        "args": {
                            "evidence_ids": ["e1", "e2", "e3", "e4", "ghost-1"],
                            "reason": "写作需要",
                        },
                        "id": "r1",
                    }
                ],
            )
        if read_calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "ReadEvidence",
                        "args": {"evidence_ids": ["e3", "e4"], "reason": "补齐截断"},
                        "id": "r2",
                    }
                ],
            )
        report = (
            "## 一\n\n甲事实。[[cite:e1]] 乙事实。[[cite:e2]] "
            "丙事实。[[cite:e3]] 丁事实。[[cite:e4]]"
        )
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "CompleteReport",
                    "args": {"selected_evidence_ids": ["e1", "e2", "e3", "e4"], "markdown": report},
                    "id": "c1",
                }
            ],
        )


def test_read_evidence_truncation_is_explicit_and_recoverable(tmp_path):
    """批量截断必须显式回传；编造 id 一次性报全且不吞配额（run-e10d1229 回归锁）。"""
    import json

    from deepsearch_agent.schemas import WriterDirective
    from fakes import make_evidence

    sink = JsonlSink(tmp_path / "events.jsonl")
    state = {
        "query": "测试",
        "evidences": [make_evidence(f"e{i}", f"事实{i}") for i in range(1, 5)],
        "writer_directive": WriterDirective(
            query="测试",
            report_brief=REPORT_BRIEF,
            research_status="completed",
            generation_mode="full",
        ),
    }
    writer = ReportWriter(
        _BulkReadThenWriteLLM(),
        AgentConfig(writer_read_batch_size=2),
        render_incomplete=lambda _s: "incomplete",
        event_sink=sink,
    )
    result = asyncio.run(writer.run(state))

    records = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    reads = [r for r in records if r["event_type"] == "writer_evidence_read"]
    assert reads[0]["payload"]["truncated_ids"] == ["e3", "e4"]  # 截断显式
    assert reads[0]["payload"]["unknown_ids"] == ["ghost-1"]  # 编造 id 被点名
    assert reads[0]["payload"]["read_ids"] == ["e1", "e2"]  # 未知 id 不占配额
    assert result["writer"].status == "completed"  # 两步内自愈
