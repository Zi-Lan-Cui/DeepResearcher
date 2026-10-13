"""最终报告渲染节点。"""

from deepresearcher.reporting import render_final_report, render_incomplete_report
from deepresearcher.schemas import (
    RenderOutcome,
    ReviewProgress,
    RunStatus,
    SupervisorProgress,
    WriterProgress,
    terminal_reason_text,
)
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
    if state.get("answer_mode") == "quick_answer":
        return {
            "report": report,
            "run": RunStatus(phase="completed", terminal_reason=RenderOutcome.QUICK_ANSWER),
        }
    if writer.status in {"failed", "exhausted"}:
        feedback = str(writer.feedback or "报告草稿未能通过引用协议校验。")
        return {
            "report": render_incomplete_report(state, [f"报告写作未能完成：{feedback}"]),
            "answer_mode": "research_incomplete",
            "run": RunStatus(phase="completed", terminal_reason=RenderOutcome.WRITER_EXHAUSTED),
        }
    if research.status == "incomplete" and not state.get("report_draft"):
        feedback = terminal_reason_text(run.terminal_reason)
        return {
            "report": render_incomplete_report(state, [feedback]),
            "answer_mode": "research_incomplete",
            "run": RunStatus(phase="completed", terminal_reason=RenderOutcome.RESEARCH_INCOMPLETE),
        }
    if state.get("answer_mode") == "research_incomplete":
        evidence_count = len(state.get("evidences", []))
        message = (
            "当前未取得足以支撑事实性结论的可验证证据。"
            if evidence_count == 0
            else f"当前已收集 {evidence_count} 条证据，但未形成可交付的完整报告。"
        )
        return {
            "report": report + f"\n\n> 注：{message}",
            "run": RunStatus(phase="completed", terminal_reason=RenderOutcome.RESEARCH_INCOMPLETE),
        }
    if run.terminal_reason == RenderOutcome.REVIEW_RECOVERY_EXHAUSTED:
        citations = list(state.get("citations", []))
        draft = state.get("report_draft", "")
        if citations and draft:
            report = render_final_report(
                body=draft,
                citations=citations,
                title=str(state.get("report_title") or ""),
            )
            return {
                "report": report + "\n\n> 注：审阅修订已达上限，本报告按最后一版交付。",
                "answer_mode": "review_limited",
                "run": RunStatus(
                    phase="completed", terminal_reason=RenderOutcome.REVIEW_RECOVERY_EXHAUSTED
                ),
            }
    if review.status == "rejected":
        feedback = review.feedback or "整体审阅未通过。"
        return {
            "report": render_incomplete_report(state, [f"整体审阅未通过：{feedback}"]),
            "answer_mode": "research_incomplete",
            "run": RunStatus(phase="completed", terminal_reason=RenderOutcome.REVIEW_REJECTED),
        }

    citations = list(state.get("citations", []))
    draft = state.get("report_draft", "")
    if not citations or not draft:
        return {
            "report": render_incomplete_report(
                state,
                ["报告缺少草稿或引用元数据，无法完成最终渲染。"],
            ),
            "answer_mode": "research_incomplete",
            "run": RunStatus(phase="completed", terminal_reason=RenderOutcome.MISSING_REPORT_DRAFT),
        }
    return {
        "report": render_final_report(
            body=draft,
            citations=citations,
            title=str(state.get("report_title") or ""),
        ),
        "run": RunStatus(phase="completed", terminal_reason=RenderOutcome.REPORT_RENDERED),
    }
