"""最终报告渲染节点。"""

from deepresearcher.reporting import render_final_report, render_incomplete_report
from deepresearcher.schemas import ReviewProgress, RunStatus, SupervisorProgress, WriterProgress
from deepresearcher.state import ResearchState, section


async def render_final_report_node(state: ResearchState):
    """流水线终点：统一处理成功、澄清和各类失败路径。"""
    run = section(state, "run", RunStatus)
    research = section(state, "supervisor", SupervisorProgress)
    writer = section(state, "writer", WriterProgress)
    review = section(state, "review", ReviewProgress)
    report = state.get("report", "")
    if run.phase == "failed":
        return {"report": report, "run": run}
    if state.get("answer_mode") == "clarification_needed":
        return {
            "report": report,
            "run": RunStatus(phase="completed", terminal_reason="clarification_needed"),
        }
    if state.get("answer_mode") == "quick_answer":
        return {
            "report": report,
            "run": RunStatus(phase="completed", terminal_reason="quick_answer"),
        }
    if writer.status in {"failed", "exhausted"}:
        feedback = str(writer.feedback or "报告草稿未能通过引用协议校验。")
        return {
            "report": render_incomplete_report(
                state,
                [f"报告写作未能完成：{feedback}"],
            )
            + "\n\n[写作校验：未通过]",
            "answer_mode": "research_incomplete",
            "run": RunStatus(phase="completed", terminal_reason="writer_exhausted"),
        }
    if research.status == "incomplete" and not state.get("report_draft"):
        feedback = str(run.terminal_reason or "研究未完成。")
        return {
            "report": render_incomplete_report(
                state,
                [f"研究阶段未完成：{feedback}"],
            )
            + "\n\n[研究阶段：未完成；未启动 Writer]",
            "answer_mode": "research_incomplete",
            "run": RunStatus(phase="completed", terminal_reason="research_incomplete"),
        }
    if state.get("answer_mode") == "research_incomplete":
        evidence_count = len(state.get("evidences", []))
        message = (
            "当前未取得足以支撑事实性结论的可验证 Evidence。"
            if evidence_count == 0
            else f"当前已收集 {evidence_count} 条 Evidence，但未形成可交付的完整报告。"
        )
        return {
            "report": report + f"\n\n[研究未完成：{message}]",
            "run": RunStatus(phase="completed", terminal_reason="research_incomplete"),
        }
    if run.terminal_reason == "review_recovery_exhausted":
        citations = list(state.get("citations", []))
        draft = state.get("report_draft", "")
        if citations and draft:
            report = render_final_report(
                clarified_query=str(state.get("clarified_query", state.get("query", ""))),
                current_round=research.current_round,
                evidence_count=int(state.get("evidence_count", len(state.get("evidences", [])))),
                body=draft,
                citations=citations,
            )
            return {
                "report": report + "\n\n[审阅状态：已达到修订上限，按最后一版交付]",
                "answer_mode": "review_limited",
                "run": RunStatus(phase="completed", terminal_reason="review_recovery_exhausted"),
            }
    if review.status == "rejected":
        feedback = review.feedback or "整体审阅未通过。"
        return {
            "report": render_incomplete_report(state, [feedback]) + "\n\n[整体审阅：未通过]",
            "answer_mode": "research_incomplete",
            "run": RunStatus(phase="completed", terminal_reason="review_rejected"),
        }

    citations = list(state.get("citations", []))
    draft = state.get("report_draft", "")
    if not citations or not draft:
        return {
            "report": render_incomplete_report(
                state,
                ["报告缺少 Writer 产出的草稿或引用元数据，无法渲染最终报告。"],
            )
            + "\n\n[渲染校验：未通过]",
            "answer_mode": "research_incomplete",
            "run": RunStatus(phase="completed", terminal_reason="missing_report_draft"),
        }
    return {
        "report": render_final_report(
            clarified_query=str(state.get("clarified_query", state.get("query", ""))),
            current_round=research.current_round,
            evidence_count=int(state.get("evidence_count", len(state.get("evidences", [])))),
            body=draft,
            citations=citations,
        ),
        "run": RunStatus(phase="completed", terminal_reason="report_rendered"),
    }
