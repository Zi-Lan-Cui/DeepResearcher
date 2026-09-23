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
    assert "Supervisor 报告任务书" in captured["context"]
    assert '"support": "direct"' in captured["context"]
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


def _decision():
    return ReviewDecision(feedback="通过", gaps=[], issues=[])


def _review_state():
    return {
        "clarified_query": "q",
        "writer_directive": _review_directive(),
        "review_attempts": 0,
        "paragraph_bindings": [_binding("A 文学性高", ["e1"], kind="evidence")],
        "citations": [_cite("e1", claim="A 有复杂叙事", quote="A 有复杂叙事。")],
    }


def test_reviewer_retries_content_filter_then_succeeds(monkeypatch):
    from openai import ContentFilterFinishReasonError

    monkeypatch.setattr("asyncio.sleep", _no_sleep)
    calls = {"n": 0}

    async def flaky(llm, schema, messages):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ContentFilterFinishReasonError()
        return _decision()

    result = asyncio.run(
        reviewer_core(
            _review_state(), object(), agent_config=AgentConfig(), invoke_structured=flaky
        )
    )
    assert result["review"].status == "approved"
    assert calls["n"] == 2  # 首次被内容审查拦截，重试一次成功


def test_reviewer_exhausts_retries_and_reraises(monkeypatch):
    from openai import ContentFilterFinishReasonError

    monkeypatch.setattr("asyncio.sleep", _no_sleep)
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
        assert False, "应当抛出"
    except ContentFilterFinishReasonError:
        pass
    # attempts=1 → 共 2 次尝试后放弃
    from deepresearcher.config import get_settings

    assert calls["n"] == get_settings().agent.reviewer_retry_attempts + 1


async def _no_sleep(_seconds):
    return None
