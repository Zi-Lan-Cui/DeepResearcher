"""Supervisor 方向派发业务实现:不感知 langgraph 的纯 async 函数。

delegate_research 执行一次 ResearchDelegate 工具请求:预算 hard check、任务
编号/去重、派发 ResearchAgent、吸收结果,返回给模型的完整方向报告 dict;
工具回执与消息配对留在协议层(tools.py),本模块不认识 ToolMessage。
簿记并发由 bookkeeping_lock 保护,实际子 Agent 并发受 deps.worker_limit 限制。
"""

import asyncio

from deepresearcher.agents.supervisor.state import (
    SupervisorDeps,
    SupervisorLoopState,
    TaskExecution,
    evidence_card,
)
from deepresearcher.context.execution import AgentExecutionScope
from deepresearcher.schemas import ResearchAgentResult, StopReason
from deepresearcher.state import SubTask


async def delegate_research(
    deps: SupervisorDeps,
    loop_state: SupervisorLoopState,
    scope: AgentExecutionScope,
    bookkeeping_lock: asyncio.Lock,
    topic: str,
) -> dict[str, object]:
    """执行一次 ResearchDelegate 工具请求:预算 hard check、任务编号/去重、
    派发 ResearchAgent、吸收结果、返回完整方向报告。
    """
    round_no = loop_state.current_round

    def reported(result: dict[str, object]) -> dict[str, object]:
        # 规划器的工具调用若被静默消化（blocked/skipped），事件流里只会
        # 看到连续两个 model_turn——delegate_started/completed 让“空轮次”可解释。
        deps.emit(
            "delegate_completed",
            {
                "status": str(result.get("status", "")),
                "reason": str(result.get("reason", "")),
                "topic_chars": len(topic),
                "evidence_count": result.get("evidence_count"),
                "source_count": result.get("source_count"),
            },
        )
        return result

    deps.emit("delegate_started", {"topic_chars": len(topic)})
    if round_no > deps.config.max_research_rounds:
        loop_state.set_stop_reason(StopReason.GLOBAL_ROUND_BUDGET_EXHAUSTED)
        return reported(
            {
                "status": "blocked",
                "reason": "round_budget_exhausted",
                "instruction": "研究轮次预算已耗尽；请修订最新研究综合稿。若达到完整标准则调用 ResearchComplete，否则直接结束，系统将按 partial 交付。",
            }
        )
    async with bookkeeping_lock:
        task_index = loop_state.allocate_task_index()
        task: SubTask = {
            "id": f"task-{task_index:04d}",
            "run_id": scope.run_id,
            "question": topic,
            "round": round_no,
            "sequence": task_index,
            "type": "search",
            "status": "pending",
            "assigned_agent": "research_agent",
            "worker_id": f"research-agent-{task_index:04d}",
            "worker_index": task_index,
            "parent_task_id": "",
            "operation_id": f"research-task-{task_index:04d}",
        }
        new_tasks = loop_state.filter_new_tasks(
            [task], max_tasks=deps.config.max_subtasks_per_round
        )
    if not new_tasks:
        loop_state.set_stop_reason(StopReason.NO_NEW_TASKS)
        return reported({"status": "skipped", "reason": "duplicate_or_budget", "topic": topic})
    execution = await execute_research_task(deps, new_tasks[0])
    async with bookkeeping_lock:
        loop_state.absorb(execution)
    direction_report: dict[str, object] = {
        "status": execution.task_result.execution_status,
        "research_direction": execution.task_result.research_direction,
        "coverage_status": execution.task_result.coverage_status,
        "evidence_count": execution.task_result.evidence_count,
        "source_count": execution.task_result.source_count,
        "remaining_gaps": execution.task_result.remaining_gaps,
        "conclusion": execution.task_result.conclusion,
        "failures": execution.task_result.failures,
        "working_set_revision": loop_state.working_set_revision,
        "evidence": [
            evidence_card(item)
            for item in execution.evidences
            if item.evidence_id in loop_state.active_evidence_ids
        ],
    }
    if execution.task_result.provider_exhausted:
        # 数据提示（无分支控制流）：让 Supervisor 模型读到系统性不可用后自然停止派发、收尾。
        direction_report["provider_exhausted"] = True
        direction_report["instruction"] = (
            "搜索服务账户级不可用（额度耗尽/密钥无效），系统性问题：再派新方向也会同样失败。"
            "停止派发 ResearchDelegate；把已有 Evidence 修订进综合稿，随后 "
            "ResearchComplete（足以成文）或 ResearchReady（保存部分报告）收尾。"
        )
    return reported(direction_report)


async def execute_research_task(deps: SupervisorDeps, task: SubTask) -> TaskExecution:
    """执行单个方向研究；worker 异常降级为 failed 结果，不中断整轮。"""
    round_no = int(task.get("round", 1))
    task_context = {
        "task_id": task["id"],
        "worker_id": task.get("worker_id", task["id"]),
        "worker_index": int(task.get("worker_index", task.get("sequence", 0))),
        "parent_task_id": task.get("parent_task_id", ""),
        "operation_id": task.get("operation_id", task["id"]),
        "concurrency_limit": deps.config.max_parallel_workers,
    }
    async with deps.worker_limit:
        deps.emit(
            "research_task_started",
            {
                **task_context,
                "task_index": int(task.get("sequence", 0)),
                "component": "research_agent",
                "question": task["question"][: deps.config.supervisor_preview_chars],
                "type": task["type"],
            },
        )
        try:
            result = await deps.research_agent.run(task)
            # 研究员是子 Agent 边界：在写入审计事件或 State 前先验证结果契约；
            # 同 run 内已是模型时零开销，防御性恢复覆盖跨进程 checkpoint。
            agent_result = (
                result
                if isinstance(result, ResearchAgentResult)
                else ResearchAgentResult.model_validate(result)
            )
            task_result = agent_result.task_result
            evidences = list(agent_result.evidences)
            selected_ids = list(agent_result.selected_evidence_ids)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            execution = TaskExecution.failed_for(
                task,
                round_no,
                str(exc)[: deps.config.supervisor_preview_chars],
            )
            deps.emit(
                "research_task_failed",
                {**task_context, **execution.task_result.model_dump()},
                component="research_agent",
            )
            return execution
        deps.emit(
            "research_task_completed",
            {**task_context, **task_result.model_dump()},
            component="research_agent",
        )
        return TaskExecution(
            task_result=task_result,
            evidences=evidences,
            selected_evidence_ids=selected_ids,
            source_refs=list(agent_result.source_refs),
        )
