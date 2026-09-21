"""Supervisor 的标准工具注册表。"""

from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool, tool
from langgraph.graph import END
from langgraph.types import Command
from pydantic import ValidationError

from deepresearcher.agents.supervisor import services
from deepresearcher.agents.supervisor.state import (
    SupervisorLoopContext,
    synthesis_snapshot,
    working_set_snapshot,
)
from deepresearcher.schemas import (
    ReadWorkingSet,
    ReleaseEvidence,
    ResearchAspect,
    ResearchComplete,
    ResearchDelegate,
    ResearchSynthesis,
    RestoreEvidence,
    ReviseResearchSynthesis,
    StopReason,
    format_tool_receipt,
)


def build_supervisor_tools() -> list[BaseTool]:
    """创建绑定到 SupervisorLoopContext 的工具集合。"""

    @tool("ResearchDelegate", args_schema=ResearchDelegate)
    async def research_delegate(
        research_topic: str,
        runtime: ToolRuntime[SupervisorLoopContext],
    ) -> str:
        """派发一个具体、可验证且与历史互补的研究方向。"""
        ctx = runtime.context
        result = await services.delegate_research(
            ctx.deps, ctx.loop_state, ctx.scope, ctx.bookkeeping_lock, research_topic
        )
        return format_tool_receipt(result)

    # return_direct 才会让 Command(goto=END) 真正终止 Agent 循环；
    # 缺省时 langchain 仍会把消息送回模型，决策调用白白空转一整圈。
    @tool("ResearchComplete", args_schema=ResearchComplete, return_direct=True)
    async def research_complete(
        synthesis_revision: int,
        reason: str,
        runtime: ToolRuntime[SupervisorLoopContext],
    ) -> Command:
        """冻结最新且未过期的研究综合稿，并终止研究阶段。"""
        loop_state = runtime.context.loop_state
        synthesis = loop_state.research_synthesis
        accepted = bool(
            synthesis is not None
            and synthesis.revision == synthesis_revision
            and loop_state.synthesis_is_fresh(synthesis)
            and synthesis.selected_evidence_ids
        )
        if accepted:
            assert synthesis is not None  # accepted 已包含该条件，供静态类型收窄。
            loop_state.sufficient = (
                all(
                    not aspect.required or aspect.status == "covered"
                    for aspect in synthesis.aspects
                )
                and not synthesis.open_gaps
                and not synthesis.conflicts
            )
            loop_state.completed_synthesis = synthesis
            loop_state.set_stop_reason(
                StopReason.SUFFICIENT if loop_state.sufficient else StopReason.SUBMITTED_WITH_GAPS
            )
        else:
            loop_state.failure_details.append(
                "ResearchComplete 拒绝了过期、缺失或未绑定 Evidence 的研究综合稿。"
            )
        return Command(
            goto=END,
            update={
                "messages": [
                    {
                        "role": "tool",
                        "content": format_tool_receipt(
                            {
                                "status": "accepted" if accepted else "rejected",
                                "reason": reason,
                                "requested_revision": synthesis_revision,
                                "current_working_set_revision": loop_state.working_set_revision,
                                **synthesis_snapshot(synthesis),
                            }
                        ),
                        "name": "ResearchComplete",
                        "tool_call_id": runtime.tool_call_id,
                    }
                ]
            },
        )

    @tool("ReviseResearchSynthesis", args_schema=ReviseResearchSynthesis)
    async def revise_research_synthesis(
        expected_revision: int,
        expected_working_set_revision: int,
        answer_goal: str,
        overall_summary: str,
        aspects: list[ResearchAspect],
        open_gaps: list[str],
        conflicts: list[str],
        next_actions: list[str],
        decision_rationale: str,
        runtime: ToolRuntime[SupervisorLoopContext],
    ) -> str:
        """用最新方向结果修订当前唯一的研究综合稿；不会结束研究。

        仅当新的研究结果实质改变结论、Evidence 选择、缺口、冲突、下一步或
        结论状态时使用。它不是工具日志；所有事实总结必须绑定当前活跃
        Evidence。aspects 是跨 Researcher 方向的认知组织单元，系统会从其
        evidence_ids 稳定推导总选择集。调用 ResearchComplete 前必须先让本综合稿
        对齐最新 working_set_revision。
        """
        loop_state = runtime.context.loop_state
        current_revision = (
            loop_state.research_synthesis.revision if loop_state.research_synthesis else 0
        )
        if (
            expected_revision != current_revision
            or expected_working_set_revision != loop_state.working_set_revision
        ):
            return format_tool_receipt(
                {
                    "status": "stale",
                    "expected_revision": current_revision,
                    "expected_working_set_revision": loop_state.working_set_revision,
                }
            )
        active_ids = set(loop_state.active_evidence_ids)
        referenced_ids = {evidence_id for aspect in aspects for evidence_id in aspect.evidence_ids}
        unknown_ids = sorted(referenced_ids - active_ids)
        if unknown_ids:
            return format_tool_receipt(
                {
                    "status": "rejected",
                    "reason": "研究综合稿只能引用当前活跃 Evidence。",
                    "invalid_evidence_ids": unknown_ids,
                    **working_set_snapshot(loop_state),
                }
            )
        try:
            synthesis = ResearchSynthesis(
                revision=current_revision + 1,
                based_on_working_set_revision=loop_state.working_set_revision,
                answer_goal=answer_goal,
                overall_summary=overall_summary,
                aspects=aspects,
                open_gaps=open_gaps,
                conflicts=conflicts,
                next_actions=next_actions,
                decision_rationale=decision_rationale,
            )
        except ValidationError as exc:
            issues = [
                {
                    "field": ".".join(str(part) for part in error["loc"]),
                    "message": error["msg"],
                }
                for error in exc.errors(include_url=False, include_input=False)
            ]
            return format_tool_receipt(
                {
                    "status": "rejected",
                    "reason": "研究综合稿不满足提交契约，请按 issues 修正后重试。",
                    "issues": issues,
                    **working_set_snapshot(loop_state),
                }
            )
        loop_state.research_synthesis = synthesis
        assigned_ids = set(synthesis.selected_evidence_ids)
        return format_tool_receipt(
            {
                "status": "accepted",
                **synthesis_snapshot(synthesis),
                "unassigned_active_evidence_ids": [
                    item.evidence_id
                    for item in loop_state.active_evidences()
                    if item.evidence_id not in assigned_ids
                ],
            }
        )

    @tool("ReadWorkingSet", args_schema=ReadWorkingSet)
    async def read_working_set(
        reason: str,
        runtime: ToolRuntime[SupervisorLoopContext],
    ) -> str:
        """查看当前活跃 Evidence 的轻量摘要。"""
        del reason
        return format_tool_receipt(working_set_snapshot(runtime.context.loop_state))

    @tool("ReleaseEvidence", args_schema=ReleaseEvidence)
    async def release_evidence(
        evidence_ids: list[str],
        reason: str,
        runtime: ToolRuntime[SupervisorLoopContext],
    ) -> str:
        """释放当前工作集中的 Evidence，但不删除全局 Evidence。"""
        del reason
        loop_state = runtime.context.loop_state
        released = loop_state.release_evidence(evidence_ids)
        return format_tool_receipt(
            {
                "released_evidence_ids": released,
                "unknown_evidence_ids": [item for item in evidence_ids if item not in released],
                **working_set_snapshot(loop_state),
            }
        )

    @tool("RestoreEvidence", args_schema=RestoreEvidence)
    async def restore_evidence(
        evidence_ids: list[str],
        reason: str,
        runtime: ToolRuntime[SupervisorLoopContext],
    ) -> str:
        """从全局 Evidence 档案恢复候选，不超过活跃工作集上限。"""
        del reason
        loop_state = runtime.context.loop_state
        restored = loop_state.restore_evidence(evidence_ids)
        return format_tool_receipt(
            {
                "restored_evidence_ids": restored,
                "not_restored_evidence_ids": [
                    item for item in evidence_ids if item not in restored
                ],
                **working_set_snapshot(loop_state),
            }
        )

    return [
        research_delegate,
        revise_research_synthesis,
        research_complete,
        read_working_set,
        release_evidence,
        restore_evidence,
    ]
