"""Supervisor 的标准工具注册表。"""

from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool, tool
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

    # 不设 return_direct、不用 Command 做循环控制(实测 Command(goto) 会跳过中间件
    # 流水线,反复拒绝时撞 recursion wall):两个分支都回 plain 回执,默认边天然
    # 把拒绝送回模型自愈;接受后由 SubmittedExitMiddleware 的 jump_to 在下一跳出环。
    @tool("ResearchComplete", args_schema=ResearchComplete)
    async def research_complete(
        synthesis_revision: int,
        reason: str,
        runtime: ToolRuntime[SupervisorLoopContext],
    ) -> str:
        """冻结最新且未过期的研究综合稿；被拒则按回执修正后重提。"""
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
        # 拒绝回执自带 requested vs current 两个 revision 与综合稿快照,模型同回合
        # 修正后重提即可;不写 failure_details——自愈的协议失误不是执行故障,
        # 反复不成时终局归属模型调用天花板。接受时 completed_synthesis 已落盘,
        # SubmittedExitMiddleware 在下一跳静默出环。
        return format_tool_receipt(
            {
                "status": "accepted" if accepted else "rejected",
                "reason": reason,
                "requested_revision": synthesis_revision,
                "current_working_set_revision": loop_state.working_set_revision,
                **synthesis_snapshot(synthesis),
            }
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
        archived = {item.evidence_id for item in loop_state.evidences}
        return format_tool_receipt(
            {
                "released_evidence_ids": released,
                # 重复释放同一 id ≠ 编造:档案在而工作集无,单列一键;
                # unknown 只留给真不在档案的 id——与 Restore 的两键形状对齐。
                "not_in_working_set_ids": [
                    item for item in evidence_ids if item in archived and item not in released
                ],
                "unknown_evidence_ids": [item for item in evidence_ids if item not in archived],
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
