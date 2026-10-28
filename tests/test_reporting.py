import asyncio

import pytest

from deepresearcher.nodes import (
    render_final_report_node,
)
from deepresearcher.reporting import (
    no_evidence_blockers,
    render_final_report,
    render_incomplete_report,
    validate_and_bind,
)
from deepresearcher.reporting.validation import DraftProtocolError
from deepresearcher.schemas import (
    Citation,
    SupervisorProgress,
)
from fakes import make_evidence as _ev


def test_render_final_report_renumbers_by_first_appearance_and_keeps_quotes():
    report = render_final_report(
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

    # 编号按正文首现顺序：e2 → 证据1，e1 → 证据2
    assert "[证据1]" in report
    assert "[证据2]" in report
    assert report.index("[证据1]") < report.index("[证据2]")
    assert "## 证据来源" in report
    # 来源表逐条保留 quote，交付报告不丢原文片段
    assert "「第二条。」" in report
    assert "「第一条。」" in report
    assert "https://example.com/two" in report
    assert "https://example.com/one" in report


def test_render_final_report_collapses_multiline_title_and_quote():
    """标题/引文内嵌换行必须合并为单行：来源表按行消费，前端逐行解析，
    换行会把一条记录劈断导致整条失配（实场 Adams Historical Returns 无来源事故）。"""
    report = render_final_report(
        body="正文。[[cite:e1]]",
        citations=[
            Citation(
                id="e1",
                url="https://example.com/one",
                title="Historical \n    Returns on Stocks: 1928-2024",
                quote="多行\n  引文\n片段",
                claim="结论",
            )
        ],
    )
    reference_lines = [line for line in report.splitlines() if line.startswith("- [证据")]
    assert len(reference_lines) == 1
    assert "Historical Returns on Stocks: 1928-2024: https://example.com/one" in reference_lines[0]
    quote_lines = [line for line in report.splitlines() if line.strip().startswith("> 「")]
    assert quote_lines == ["  > 「多行 引文 片段」"]


def test_render_final_report_strips_writer_authored_reference_section():
    """Writer 自写来源小节时由程序剥除，参考表以管道追加的一段为准。

    剥除在编号渲染之前：被剥段里的 [[cite:e1]] 不参与首现顺序，仅出现在该段的引用不进参考表。
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
        body=(
            "## 结论\n\n龙的形象综合。[[cite:e2]]\n\n"
            "### 主要参考\n- 来源二：[[cite:e2]]\n- 来源一：[[cite:e1]]\n"
        ),
        citations=citations,
    )
    assert report.count("## 证据来源") == 1  # 只剩管道追加的那份
    assert "主要参考" not in report
    assert "### " not in report  # 被剥小节的标题也不残留
    assert "「第二条。」" in report  # 正文标记正常入表
    assert "「第一条。」" not in report  # 只在被剥小节出现的引用被剔除
    assert "https://example.com/one" not in report


def test_render_final_report_strip_stops_at_next_same_level_heading():
    report = render_final_report(
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
    assert "这里还要引用" in report  # 剥段范围越界会吃掉后文
    assert report.count("## 证据来源") == 1  # 正文的 ## 参考来源 被剥，只剩管道追加的表
    # e1 在存活正文中再次出现仍入表，e2 首现更早故 e1 编号为 2
    assert "「第一条。」" in report
    assert report.index("前言") < report.index("这里还要引用")


def test_render_final_report_ignores_cite_markers_inside_code():
    report = render_final_report(
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
    assert "[证据1]" in report
    assert "https://example.com/two" not in report  # 代码里的伪标记不进入证据来源表


def test_render_final_report_node_routes_failure_paths():
    # 写作失败 → 降级渲染
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

    # 审阅拒绝且恢复耗尽 → 降级渲染
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
    assert "## 证据来源" in result["report"]
    assert "审阅修订已达上限" in result["report"]

    # 审阅弃权 → 草稿齐备照常交付，但标明未经审阅
    result = asyncio.run(
        render_final_report_node(
            {
                "clarified_query": "A 的性能",
                "run": {"phase": "reviewing"},
                "review": {"status": "abstained", "attempts": 3},
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
    assert result["run"].terminal_reason == "review_abstained"
    assert "未经审阅" in result["report"]
    assert "## 证据来源" in result["report"]

    # 正常路径：草稿 + citations → 渲染
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
    assert "[证据1]" in result["report"]
    assert "## 证据来源" in result["report"]
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
        body="第一条结论。[[cite:e1]]\n\n第二条结论。[[cite:e2]]",
        citations=citations,
    )

    assert report.index("[证据1]") < report.index("[证据2]")
    assert "「第一条原文。」" in report
    assert "「第二条原文。」" in report
    assert "来源一: https://one.test" in report
    assert "来源二: https://two.test" in report


def test_render_never_leaks_machine_terminal_reason_into_report():
    """StopReason 值不得原文进入用户报告,须译为中文文案。"""
    from deepresearcher.nodes.render import render_final_report_node
    from deepresearcher.schemas import RunStatus, SupervisorProgress

    state = {
        "query": "q",
        "run": RunStatus(phase="rendering", terminal_reason="ended_without_completion"),
        "supervisor": SupervisorProgress(status="incomplete"),
    }
    result = asyncio.run(render_final_report_node(state))
    assert "ended_without_completion" not in result["report"]
    assert "模型结束研究而未显式收尾" in result["report"]


def test_reference_table_survives_ghost_cite():
    """正文引用无元数据的 id 时降级为编号显示,终检不得抛 KeyError。"""
    from deepresearcher.reporting import render_final_report

    report = render_final_report(
        body="事实一句。[[cite:e1]] 幽灵句。[[cite:ghost]]",
        citations=[
            Citation(id="e1", url="https://example.com/a", quote="事实一句。", claim="事实一句")
        ],
    )
    assert "证据来源" in report
    assert "证据1" in report
    assert "证据2" in report  # 无元数据的引用仍占编号位


def test_final_report_is_article_shaped_with_model_title():
    report = render_final_report(
        body="结论句。",
        citations=[],
        title="A 的真相",
    )
    assert report.startswith("# A 的真相\n")
    assert "## 研究问题" not in report  # 不输出系统结构小节
    assert "研究问题" not in report  # 问题与统计由页面展示，不写入报告正文
    assert "## 证据来源" in report


def test_final_report_without_title_falls_back_to_system_heading():
    report = render_final_report(body="正文", citations=[])
    assert report.startswith("# 研究报告\n")


def test_incomplete_report_collapses_zero_evidence_to_diagnostic_card():
    """零证据时塌缩为诊断卡:只给原因与恢复方式,不逐条罗列方向和缺口。"""
    state = {
        "clarified_query": "如何克服恐惧",
        "evidences": [],
        "supervisor": SupervisorProgress(current_round=1),
        "task_results": [
            {
                "execution_status": "completed",
                "coverage_status": "insufficient",
                "research_direction": "梳理舒适圈概念的心理学理论基础，要求引用一手学术文献并标注出处",
                "evidence_count": 0,
                "provider_exhausted": True,
            }
        ],
    }

    report = render_incomplete_report(state, ["搜索服务额度耗尽"])

    assert "搜索服务额度耗尽" in report
    assert "系统性故障" in report  # provider_exhausted 命中，给系统性说明
    assert "恢复方式" in report
    assert "任务结果" not in report  # 塌缩为一段，不逐条罗列
    assert "未闭合缺口" not in report


def test_partial_report_lists_task_titles_without_direction_conclusions():
    """有证据但部分交付:只列方向短题与状态,不复述方向结论——报告只认 Supervisor 的任务书。"""
    from types import SimpleNamespace

    long_direction = "研究人类面对未知时产生恐惧的神经机制" + "，覆盖杏仁核与前额叶" * 8
    state = {
        "clarified_query": "如何克服恐惧",
        "evidences": [SimpleNamespace(source_url=f"https://a/{i}") for i in range(3)],
        "supervisor": SupervisorProgress(current_round=1),
        "task_results": [
            {
                "execution_status": "completed",
                "coverage_status": "partial",
                "research_direction": long_direction,
                "evidence_count": 3,
                "conclusion": "SECRET-方向结论",
            }
        ],
    }

    report = render_incomplete_report(state, ["预算耗尽"])

    assert "任务结果" in report
    assert "…" in report  # 超长课题被截断，不逐字回声
    assert "SECRET-方向结论" not in report  # 方向结论不进报告
    assert "未闭合缺口" not in report  # 缺口已并入方向结论，不单列
