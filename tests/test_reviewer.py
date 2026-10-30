import asyncio

from deepresearcher.config import AgentConfig
from deepresearcher.nodes import (
    reviewer,
)
from deepresearcher.nodes.reviewer import reviewer as reviewer_core
from deepresearcher.schemas import (
    ReportBrief,
    ReviewDecision,
    ReviewIssue,
    WriterDirective,
)
from fakes import REPORT_BRIEF
from fakes import make_binding as _binding
from fakes import make_citation as _cite


def _review_directive(query: str = "q") -> WriterDirective:
    """生产里任务书只经 writer_directive 交接。"""
    return WriterDirective(
        query=query,
        report_brief=ReportBrief.model_validate(REPORT_BRIEF),
        research_status="completed",
        generation_mode="full",
    )


def test_reviewer_rejects_with_structured_evidence_feedback(monkeypatch):
    captured = {}

    async def review_output(llm, schema, messages):
        assert schema is ReviewDecision
        captured["context"] = messages[-1].content
        return ReviewDecision(
            feedback="需要能直接支持文学性评价的评论来源。",
            gaps=["缺少具体作品的文学性评价依据"],
            issues=[ReviewIssue(severity="fatal", reason="核心文学性结论缺少直接来源。")],
        )

    monkeypatch.setattr("deepresearcher.nodes.ainvoke_structured", review_output)
    result = asyncio.run(
        reviewer(
            {
                "clarified_query": "哪些作品文学性高",
                "writer_directive": _review_directive("哪些作品文学性高"),
                "review_attempts": 0,
                "paragraph_bindings": [_binding("A 文学性高", ["e1"], kind="evidence")],
                "citations": [_cite("e1", claim="A 有复杂叙事", quote="A 有复杂叙事。")],
            },
            object(),
            agent_config=AgentConfig(),
        )
    )

    assert result["review"].status == "rejected"
    assert result["review"].gaps == ["缺少具体作品的文学性评价依据"]
    assert "Supervisor 报告任务书" not in captured["context"]  # 独立评审，不带上游环节
    assert '"support": "strict"' in captured["context"]
    assert '"source_domain": ""' in captured["context"]


def test_reviewer_requests_rewrite_when_evidence_is_sufficient(monkeypatch):
    async def review_output(llm, schema, messages):
        assert schema is ReviewDecision
        return ReviewDecision(
            feedback="将‘证明文学性’收窄为‘可作为获得认可的线索’。",
            gaps=[],
            issues=[ReviewIssue(severity="fatal", reason="核心结论把提名误写为文学性证明。")],
        )

    monkeypatch.setattr("deepresearcher.nodes.ainvoke_structured", review_output)
    result = asyncio.run(
        reviewer(
            {
                "clarified_query": "哪些作品文学性高",
                "review_attempts": 0,
                "paragraph_bindings": [_binding("A 的提名证明文学性", ["e1"], kind="evidence")],
                "citations": [_cite("e1", claim="A 获得提名", quote="A 获得提名。")],
            },
            object(),
            agent_config=AgentConfig(),
        )
    )

    assert result["review"].status == "rejected"
    assert result["review"].gaps == []


def test_reviewer_allows_warnings_without_rejecting_report(monkeypatch):
    async def review_output(llm, schema, messages):
        return ReviewDecision(
            summary="核心结论有来源支撑，仅一处措辞超出原文。",
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

    monkeypatch.setattr("deepresearcher.nodes.ainvoke_structured", review_output)
    result = asyncio.run(
        reviewer(
            {
                "clarified_query": "A 有何体验特点",
                "paragraph_bindings": [_binding("A 可能增强沉浸感。", ["e1"], kind="synthesis")],
                "citations": [_cite("e1", claim="A 支持 VR 漫游", quote="A 支持 VR 漫游。")],
            },
            object(),
            agent_config=AgentConfig(),
        )
    )

    assert result["review"].status == "approved"
    assert result["review"].issues[0].severity == "warning"
    # 展示结论与改稿意见分开存。
    assert result["review"].summary == "核心结论有来源支撑，仅一处措辞超出原文。"


def _review_state():
    return {
        "clarified_query": "q",
        "writer_directive": _review_directive(),
        "review_attempts": 0,
        "paragraph_bindings": [_binding("A 文学性高", ["e1"], kind="evidence")],
        "citations": [_cite("e1", claim="A 有复杂叙事", quote="A 有复杂叙事。")],
    }


def test_reviewer_sends_one_request_per_attempt(monkeypatch):
    """审阅节点单发：同一次审阅只请求一次，重发归结构化入口与模型客户端。"""
    from openai import ContentFilterFinishReasonError

    calls = {"n": 0}

    async def always_filtered(llm, schema, messages):
        calls["n"] += 1
        raise ContentFilterFinishReasonError()

    try:
        asyncio.run(
            reviewer_core(
                _review_state(),
                object(),
                agent_config=AgentConfig(),
                invoke_structured=always_filtered,
            )
        )
        raise AssertionError("应当抛出")
    except ContentFilterFinishReasonError:
        pass
    assert calls["n"] == 1


def test_reviewer_fails_when_output_length_runaways(monkeypatch):
    """输出被长度上限截断：写成 failed，不冒充审阅通过，也不重发同一请求。"""
    from types import SimpleNamespace

    from openai import LengthFinishReasonError

    calls = []

    async def runaway(llm, schema, messages):
        calls.append(1)
        raise LengthFinishReasonError(completion=SimpleNamespace(usage=None))

    monkeypatch.setattr("deepresearcher.nodes.ainvoke_structured", runaway)
    result = asyncio.run(
        reviewer(
            {
                "clarified_query": "q",
                "paragraph_bindings": [_binding("A 段", ["e1"], kind="evidence")],
                "citations": [_cite("e1", claim="事实", quote="原文。")],
            },
            object(),
            agent_config=AgentConfig(),
        )
    )

    assert len(calls) == 1
    assert result["review"].status == "failed"
    assert result["review"].issues == []
    # 失败成因写进状态：只留在日志里，详情页就只能说「未取得结论」。
    assert result["review"].failure_reason == "审阅输出超出长度上限，未返回结论"
