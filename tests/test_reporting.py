import asyncio

import pytest

from deepresearcher.orchestration.nodes import (
    render_final_report_node,
)
from deepresearcher.reporting import (
    no_evidence_blockers,
    render_final_report,
    validate_and_bind,
)
from deepresearcher.reporting.validation import DraftProtocolError
from deepresearcher.schemas import (
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


def test_render_final_report_strips_writer_authored_reference_section():
    """Writer 违令自写来源小节：程序剥除之，参考表以唯一追加段为准。

    线上龙意象运行实锤：正文一次+结尾一次两份「来源」并存。剥除必须发生在
    编号渲染之前——被剥段里的 [[cite:e1]] 不再抢占首现顺序，仅出现于此的
    citation 依既有规则不进参考表。
    """
    citations = [
        Citation(
            id="e1", url="https://example.com/one", title="来源一", quote="第一条。", claim="第一条"
        ),
        Citation(
            id="e2", url="https://example.com/two", title="来源二", quote="第二条。", claim="第二条"
        ),
    ]
    report = render_final_report(
        clarified_query="q",
        current_round=1,
        evidence_count=2,
        body=(
            "## 结论\n\n龙的形象综合。[[cite:e2]]\n\n"
            "### 主要参考\n- 来源二：[[cite:e2]]\n- 来源一：[[cite:e1]]\n"
        ),
        citations=citations,
    )
    assert report.count("## 参考来源") == 1  # 只剩管道追加的那份
    assert "主要参考" not in report
    assert "### " not in report  # 被剥小节的标题也不残留
    assert "「第二条。」" in report  # 正文标记正常入表
    assert "「第一条。」" not in report  # 只在被剥小节出现的引用被剔除
    assert "https://example.com/one" not in report


def test_render_final_report_strip_stops_at_next_same_level_heading():
    report = render_final_report(
        clarified_query="q",
        current_round=1,
        evidence_count=2,
        body=(
            "前言[[cite:e2]]\n\n## 参考来源\n- 旧[[cite:e1]]\n\n"
            "## 补充论证\n这里还要引用[[cite:e1]]收尾。"
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
    assert "这里还要引用" in report  # 误剥会吃掉后文
    assert report.count("## 参考来源") == 1
    # e1 在存活正文中重新出现 → 依然合法入表（编号 2，因 e2 首现更早）
    assert "「第一条。」" in report
    assert report.index("前言") < report.index("这里还要引用")


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

    # 审阅回流耗尽 → 保留最后一版及完整参考来源，以受限报告交付
    result = asyncio.run(
        render_final_report_node(
            {
                "clarified_query": "A 的性能",
                "run": {"phase": "rendering", "terminal_reason": "review_recovery_exhausted"},
                "review": {"status": "rejected", "attempts": 2, "feedback": "仍有缺口"},
                "supervisor": {"status": "incomplete", "current_round": 1},
                "report_draft": "## 结论\n\nA 的平均延迟为 20ms。[[cite:e1]]",
                "citations": [
                    Citation(
                        id="e1",
                        url="https://example.com/a",
                        title="来源",
                        quote="A 的平均延迟为 20ms。",
                        claim="A 的平均延迟为 20ms",
                    )
                ],
            },
        )
    )
    assert result["answer_mode"] == "review_limited"
    assert result["run"].phase == "completed"
    assert result["run"].terminal_reason == "review_recovery_exhausted"
    assert "A 的平均延迟为 20ms" in result["report"]
    assert "## 参考来源" in result["report"]
    assert "已达到修订上限" in result["report"]

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


def test_quick_answer_finalizer_does_not_append_redundant_mode_text():
    result = asyncio.run(
        render_final_report_node(
            {
                "answer_mode": "quick_answer",
                "report": "向量数据库用于存储和检索向量。",
            }
        )
    )

    assert result["report"] == "向量数据库用于存储和检索向量。"
    assert result["run"].terminal_reason == "quick_answer"


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
