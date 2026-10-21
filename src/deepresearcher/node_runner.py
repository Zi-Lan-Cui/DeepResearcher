"""顶层节点的统一运行器。

instrument_node 只记录节点生命周期；本模块把未处理的节点异常转换成
可供 LangGraph 继续推进的运行状态：失败收尾在此产出降级报告
render_incomplete_report 或 render_error_report，终点渲染节点对
failed 分支只透传。各节点因此不必各自复制一套 try/except。
"""

import asyncio
import inspect
from collections.abc import Callable
from typing import Any, cast

from langgraph.errors import GraphBubbleUp, NodeCancelledError

from deepresearcher.reporting import render_error_report, render_incomplete_report
from deepresearcher.routing import NodeName
from deepresearcher.schemas import RunError, RunStatus, failure_event_fields
from deepresearcher.state import ResearchState, restore_state_models, validate_state_invariants
from deepresearcher.usage_runtime import UsageBudgetExceeded


async def execute_node(
    state: dict[str, Any],
    *,
    stage: str,
    node: Callable[[dict[str, Any]], Any],
) -> dict[str, Any]:
    """执行一个顶层节点，并将普通异常转换为终止状态。

    取消与 KeyboardInterrupt 原样向上传播，其余异常生成 RunError 并返回最小失败状态，
    由最终渲染节点收尾。

    参数：
        state: 当前研究状态。
        stage: 节点名，用于失败事件与 RunError。
        node: 节点函数，可同步可异步。

    返回：
        节点 update，或失败时收尾 update。
    """
    try:
        restore_state_models(state)
        value = node(state)
        if inspect.isawaitable(value):
            value = await value
        result = dict(value)
        validate_state_invariants(state, result)
        return result
    except (GraphBubbleUp, asyncio.CancelledError, KeyboardInterrupt, NodeCancelledError):
        raise
    except Exception as exc:
        # 失败的生命周期事件由 observability.instrumentation 经 sink 单路记录，
        # 状态里不另存副本；此处的 error 字段与事件侧共用 failure_event_fields 单源。
        error, _ = failure_event_fields(stage, exc)
        budget_exhausted = isinstance(exc, UsageBudgetExceeded)
        if budget_exhausted:
            error = RunError(
                stage=stage,
                code="budget_exhausted",
                message="本次研究已达配置的模型用量上限。",
                retryable=False,
                detail=str(exc),
            )
        failure: dict[str, Any] = {
            "run": RunStatus(
                phase="failed",
                terminal_reason="budget_exhausted" if budget_exhausted else "node_failed",
                error=error,
            ),
            "answer_mode": "research_incomplete",
            "supervisor_next": NodeName.RENDER_FINAL_REPORT,
            "report": (
                render_incomplete_report(
                    cast(ResearchState, state),
                    ["本次研究已达用量上限，已保留当前获取的研究进度。"],
                )
                if budget_exhausted
                else render_error_report(cast(ResearchState, state), error)
            ),
        }
        # 运行器的产物与节点产物过同一条不变量校验。此处校验不通过说明
        # 错误路径本身被改坏，是运行器的 bug，直接向上抛出，不做静默降级。
        validate_state_invariants(state, failure)
        return failure
