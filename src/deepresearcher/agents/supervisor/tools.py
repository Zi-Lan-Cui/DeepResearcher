"""Supervisor 的标准工具注册表。"""

import asyncio
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.types import Command

from deepresearcher.agents.supervisor import services
from deepresearcher.agents.supervisor.state import (
    SupervisorContext,
    workbench_view,
)
from deepresearcher.schemas import (
    ReadWorkbench,
    ResearchComplete,
    ResearchDelegate,
    UpdateWorkbench,
    WaitTask,
    format_tool_receipt,
)


def _channel_command(
    runtime: ToolRuntime[SupervisorContext],
    payload: dict[str, object],
    *,
    tool_name: str,
    channel_update: dict[str, Any],
) -> Command:
    """回执与通道增量同批提交：模型读到的回执即通道将写入磁盘的状态。"""
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
    """创建绑定到 SupervisorContext 的工具集合。"""

    @tool("ResearchDelegate", args_schema=ResearchDelegate)
    async def research_delegate(
        display_title: str,
        research_topic: str,
        runtime: ToolRuntime[SupervisorContext],
    ) -> Command:
        """派发一个具体、可验证且与历史互补的研究任务；回执即受理，任务在途后台执行。"""
        ctx = runtime.context
        task, receipt, entry = services.dispatch_research(
            ctx.deps,
            ctx.round_state,
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
        if task_id in ctx.task_handles:
            # 工具重试再入同一调用：登记与任务已存在，只回受理回执，不产生第二个任务体。
            return _channel_command(
                runtime, receipt, tool_name="ResearchDelegate", channel_update={}
            )
        ctx.task_handles[task_id] = asyncio.create_task(
            services._execute_research_task(ctx.deps, task)
        )
        return _channel_command(
            runtime,
            receipt,
            tool_name="ResearchDelegate",
            channel_update={
                "pending_tasks": {"add": {task_id: entry}, "remove": []},
                # 派发编号随派发批提交：下一批从通道重建后继续前进，不再重派同号。
                "task_index_counter": ctx.round_state.task_index_counter,
            },
        )

    @tool("WaitTask", args_schema=WaitTask)
    async def wait_task(
        task_ids: list[str],
        timeout: float | None = None,
        *,
        runtime: ToolRuntime[SupervisorContext],
    ) -> Command:
        """等待在途任务完成并收取任务报告；到期无完成回 timed_out，指定 id 不在途回 unknown_task_ids，无在途回 no_inflight，都不会卡死。"""
        ctx = runtime.context
        reports, delta, timed_out = await services.wait_for_tasks(
            ctx.deps,
            ctx.round_state,
            ctx.task_handles,
            task_ids=list(task_ids or []),
            timeout=timeout,
        )
        payload = services.compose_wait_receipt(
            reports,
            set(ctx.round_state.pending_tasks),
            ctx.round_state.working_set_revision,
            timed_out=timed_out,
        )
        return _channel_command(runtime, payload, tool_name="WaitTask", channel_update=delta)

    # 不设 return_direct、不用 Command 做循环控制：Command(goto) 会跳过中间件流水线，
    # 反复拒绝时一路运行到 recursion 上限报错。两个分支都回 plain 回执，默认边把拒绝
    # 送回模型自行修正；接受后由 SubmittedExitMiddleware 的 jump_to 在下一跳退出循环。
    @tool("ResearchComplete", args_schema=ResearchComplete)
    async def research_complete(
        reason: str,
        runtime: ToolRuntime[SupervisorContext],
    ) -> Command:
        """冻结最新且已与工作台核对的研究综合稿；被拒则按回执 rejection_cause 处理。"""
        ctx = runtime.context
        result = services.freeze_synthesis(ctx.round_state, reason)
        return _channel_command(
            runtime,
            result,
            tool_name="ResearchComplete",
            channel_update={
                "completed_synthesis": ctx.round_state.completed_synthesis,
                "stop_reason": ctx.round_state.stop_reason,
            },
        )

    @tool("UpdateWorkbench", args_schema=UpdateWorkbench)
    async def update_workbench(
        changes: list[Any],
        runtime: ToolRuntime[SupervisorContext],
    ) -> Command:
        """工作台唯一写入口：提交带 op 判别字段的操作列表，每个操作整值替换其目标段。"""
        ctx = runtime.context
        result = services.apply_workbench_changes(ctx.round_state, changes)
        delta: dict[str, Any] = {}
        if result.get("status") == "applied":
            delta = {
                "research_synthesis": ctx.round_state.research_synthesis,
                "plan": [dict(item) for item in ctx.round_state.plan],
                "working_set_revision": ctx.round_state.working_set_revision,
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
        runtime: ToolRuntime[SupervisorContext],
    ) -> str:
        """查看工作台全景：计划、在途、各任务档案、活跃/储备证据与综合稿。"""
        del reason
        return format_tool_receipt(workbench_view(runtime.context.round_state))

    return [
        research_delegate,
        wait_task,
        update_workbench,
        research_complete,
        read_workbench,
    ]
