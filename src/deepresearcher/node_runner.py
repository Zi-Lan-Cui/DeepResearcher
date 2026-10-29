"""顶层节点的统一运行器"""

import asyncio
import inspect
from collections.abc import Callable
from typing import Any, cast

from langgraph.errors import GraphBubbleUp, NodeCancelledError

from deepresearcher.reporting import render_error_report
from deepresearcher.routing import NodeName
from deepresearcher.schemas import RunError, RunStatus, failure_event_fields
from deepresearcher.state import RunState, restore_state_models, validate_state_invariants
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
    except Exception as exc:  # noqa: BLE001 - 节点边界，任何异常转成 RunError 落状态
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
            "report": render_error_report(cast(RunState, state)),
        }
        # 运行器的产物与节点产物过同一条不变量校验。此处校验不通过说明
        # 错误路径本身被改坏，是运行器的 bug，直接向上抛出。
        validate_state_invariants(state, failure)
        return failure
