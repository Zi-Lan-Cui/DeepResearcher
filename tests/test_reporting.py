import asyncio

import pytest

from deepsearch_agent.orchestration.nodes import (
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
)
from fakes import make_evidence as _ev


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
        )
    )
    assert "[来源1]" in result["report"]
    assert "## 参考来源" in result["report"]
    assert "「原文。」" in result["report"]


def test_validate_and_bind_rejects_unknown_evidence_and_too_many_sources():
    with pytest.raises(DraftProtocolError, match="不存在的 Evidence"):
        validate_and_bind("事实。[[cite:unknown]]", {"e1": _ev("e1", "事实")})
    with pytest.raises(DraftProtocolError, match="最多绑定三个"):
        validate_and_bind(
            "事实。[[cite:e1,e2,e3,e4]]", {f"e{i}": _ev(f"e{i}", f"事实{i}") for i in range(1, 5)}
        )


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
