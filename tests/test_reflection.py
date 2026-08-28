import asyncio

from deepsearch_agent.orchestration.nodes import (
    reflection,
)
from deepsearch_agent.schemas import (
    ReflectionDecision,
    ReviewIssue,
)
from fakes import REPORT_BRIEF
from fakes import make_binding as _binding
from fakes import make_citation as _cite


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
