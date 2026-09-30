"""Supervisor 的标准工具注册表。"""

from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool, tool

from deepresearcher.agents.supervisor import services
from deepresearcher.agents.supervisor.state import (
    SupervisorLoopContext,
    working_set_snapshot,
)
from deepresearcher.agents.working_set import release_working_set, restore_working_set
from deepresearcher.schemas import (
    ReadWorkingSet,
    ReleaseEvidence,
    ResearchAspect,
    ResearchComplete,
    ResearchDelegate,
    RestoreEvidence,
    ReviseResearchSynthesis,
    format_tool_receipt,
)


def build_supervisor_tools() -> list[BaseTool]:
    """创建绑定到 SupervisorLoopContext 的工具集合。"""

    @tool("ResearchDelegate", args_schema=ResearchDelegate)
    async def research_delegate(
        display_title: str,
        research_topic: str,
        runtime: ToolRuntime[SupervisorLoopContext],
    ) -> str:
        """派发一个具体、可验证且与历史互补的研究方向。"""
        ctx = runtime.context
        result = await services.delegate_research(
            ctx.deps,
            ctx.loop_state,
            ctx.scope,
            ctx.bookkeeping_lock,
            research_topic,
            display_title,
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
        return format_tool_receipt(
            services.freeze_synthesis(
                runtime.context.loop_state, synthesis_revision, reason
            )
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
        return format_tool_receipt(
            services.revise_synthesis(
                runtime.context.loop_state,
                expected_revision=expected_revision,
                expected_working_set_revision=expected_working_set_revision,
                answer_goal=answer_goal,
                overall_summary=overall_summary,
                aspects=aspects,
                open_gaps=open_gaps,
                conflicts=conflicts,
                next_actions=next_actions,
                decision_rationale=decision_rationale,
            )
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
        return format_tool_receipt(
            release_working_set(
                loop_state, evidence_ids, snapshot=lambda: working_set_snapshot(loop_state)
            )
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
        return format_tool_receipt(
            restore_working_set(
                loop_state, evidence_ids, snapshot=lambda: working_set_snapshot(loop_state)
            )
        )

    return [
        research_delegate,
        revise_research_synthesis,
        research_complete,
        read_working_set,
        release_evidence,
        restore_evidence,
    ]
