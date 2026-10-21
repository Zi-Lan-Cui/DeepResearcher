"""Supervisor 的标准工具注册表。

写状态的工具在出口把回执写进 messages、业务增量写进 SupervisorAgentState 通道:并发工具
(ResearchDelegate)只交合并语义差量,串行工具(Release/Restore/Revise/Complete)交定向
的整字段快照。
"""

import asyncio
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.types import Command

from deepresearcher.agents.supervisor import services
from deepresearcher.agents.supervisor.state import (
    SupervisorLoopContext,
    workbench_view,
)
from deepresearcher.schemas import (
    ReadWorkbench,
    ResearchComplete,
    ResearchDelegate,
    UpdateWorkbench,
    WaitDirection,
    format_tool_receipt,
)


def _channel_command(
    runtime: ToolRuntime[SupervisorLoopContext],
    payload: dict[str, object],
    *,
    tool_name: str,
    channel_update: dict[str, Any],
) -> Command:
    """回执与通道增量同批提交:模型读到的回执即通道将落盘的状态。"""
    return Command(
        update={
            "messages": [
                ToolMessage(
                    format_tool_receipt(payload),
                    tool_call_id=runtime.tool_call_id,
                    name=tool_name,
                )
            ],
            **channel_update,
        }
    )


def build_supervisor_tools() -> list[BaseTool]:
    """创建绑定到 SupervisorLoopContext 的工具集合。"""

    @tool("ResearchDelegate", args_schema=ResearchDelegate)
    async def research_delegate(
        display_title: str,
        research_topic: str,
        runtime: ToolRuntime[SupervisorLoopContext],
    ) -> Command:
        """派发一个具体、可验证且与历史互补的研究方向；回执即受理，方向在途后台执行。

        任务体在 create_task 与 researcher.ainvoke 之间除并发槽外不得引入 await:
        嵌套 checkpoint ns 的后缀按 ainvoke 的调度序分配,保序才跨重放稳定。
        """
        ctx = runtime.context
        task, receipt, entry = services.dispatch_research(
            ctx.deps,
            ctx.loop_state,
            ctx.scope,
            research_topic,
            display_title,
            tool_call_id=runtime.tool_call_id or "",
            origin_state=dict(runtime.state or {}),
        )
        if task is None:
            return _channel_command(
                runtime, receipt, tool_name="ResearchDelegate", channel_update={}
            )
        task_id = str(task["id"])
        if task_id in ctx.direction_tasks:
            # 工具重试再入同一调用:登记与任务已存在,只回受理回执,不产生第二个任务体。
            return _channel_command(
                runtime, receipt, tool_name="ResearchDelegate", channel_update={}
            )
        ctx.direction_tasks[task_id] = asyncio.create_task(
            services._execute_research_task(ctx.deps, task)
        )
        return _channel_command(
            runtime,
            receipt,
            tool_name="ResearchDelegate",
            channel_update={
                "pending_directions": {"add": {task_id: entry}, "remove": []},
                # 编号水位随派发批提交:下一批从通道重建后继续前进,不再重派同号。
                "task_index_counter": ctx.loop_state.task_index_counter,
            },
        )

    @tool("WaitDirection", args_schema=WaitDirection)
    async def wait_direction(
        task_ids: list[str],
        timeout: float | None = None,
        *,
        runtime: ToolRuntime[SupervisorLoopContext],
    ) -> Command:
        """等待在途方向完成并收取方向报告；到期无完成回 timed_out，指定 id 不在途回 unknown_task_ids，无在途回 no_inflight，都不挂死。"""
        ctx = runtime.context
        reports, delta, timed_out = await services.wait_for_directions(
            ctx.deps,
            ctx.loop_state,
            ctx.direction_tasks,
            task_ids=list(task_ids or []),
            timeout=timeout,
        )
        payload = services.compose_wait_receipt(
            reports,
            set(ctx.loop_state.pending_directions),
            ctx.loop_state.working_set_revision,
            timed_out=timed_out,
        )
        return _channel_command(runtime, payload, tool_name="WaitDirection", channel_update=delta)

    # 不设 return_direct、不用 Command 做循环控制:实测 Command(goto) 会跳过中间件
    # 流水线,反复拒绝时一路运行到 recursion 上限报错。两个分支都回 plain 回执,
    # 默认边自然把拒绝送回模型自行修正;接受后由 SubmittedExitMiddleware 的 jump_to
    # 在下一跳退出循环。
    @tool("ResearchComplete", args_schema=ResearchComplete)
    async def research_complete(
        reason: str,
        runtime: ToolRuntime[SupervisorLoopContext],
    ) -> Command:
        """冻结最新且已与桌面核对的研究综合稿；被拒则按回执 rejection_cause 处理。"""
        ctx = runtime.context
        result = services.freeze_synthesis(ctx.loop_state, reason)
        return _channel_command(
            runtime,
            result,
            tool_name="ResearchComplete",
            channel_update={
                "completed_synthesis": ctx.loop_state.completed_synthesis,
                "stop_reason": ctx.loop_state.stop_reason,
            },
        )

    @tool("UpdateWorkbench", args_schema=UpdateWorkbench)
    async def update_workbench(
        changes: list[Any],
        runtime: ToolRuntime[SupervisorLoopContext],
    ) -> Command:
        """工作台唯一写入口：提交带 op 判别字段的操作列表，每个操作整值替换其目标段。

        台账是活的：方向的缺口被补齐、冲突被裁定、或某条 disclose 判定随新证据改变时，
        重提对应 aspect 改写 gaps/conflicts；同一缺陷只登记在它所属的一个板块，不在多处复制。
        """
        ctx = runtime.context
        result = services.apply_workbench_changes(ctx.loop_state, changes)
        delta: dict[str, Any] = {}
        if result.get("status") == "applied":
            delta = {
                "research_synthesis": ctx.loop_state.research_synthesis,
                "plan": [dict(item) for item in ctx.loop_state.plan],
                "working_set_revision": ctx.loop_state.working_set_revision,
            }
            removals = result.get("slot_removals")
            additions = result.get("slot_additions")
            added = [str(item) for item in additions] if isinstance(additions, list) else []
            removed = [str(item) for item in removals] if isinstance(removals, list) else []
            if added or removed:
                delta["active_evidence_ids"] = {"add": added, "remove": removed}
        return _channel_command(runtime, result, tool_name="UpdateWorkbench", channel_update=delta)

    @tool("ReadWorkbench", args_schema=ReadWorkbench)
    async def read_workbench(
        reason: str,
        runtime: ToolRuntime[SupervisorLoopContext],
    ) -> str:
        """查看工作台全景：计划、在途、各方向档案、活跃/储备证据与综合稿。"""
        del reason
        return format_tool_receipt(workbench_view(runtime.context.loop_state))

    return [
        research_delegate,
        wait_direction,
        update_workbench,
        research_complete,
        read_workbench,
    ]
