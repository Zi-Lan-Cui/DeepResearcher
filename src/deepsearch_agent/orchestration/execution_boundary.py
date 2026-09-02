"""顶层节点的统一执行边界。

观测层只记录节点生命周期；本模块负责把未处理的节点异常转换成
可供 LangGraph 继续收束的运行状态。这样错误处理不会反向依赖日志或
报告渲染实现，也不会让每个节点各自复制一套 try/except。
"""

import asyncio
import inspect
from collections.abc import Callable
from typing import Any, cast

from langgraph.errors import GraphBubbleUp, NodeCancelledError

from deepsearch_agent.observability.events.models import make_node_event
from deepsearch_agent.reporting import render_error_report, render_incomplete_report
from deepsearch_agent.routing import NodeName
from deepsearch_agent.schemas import RunError, RunLifecycle
from deepsearch_agent.service.usage import UsageBudgetExceeded
from deepsearch_agent.state import ResearchState, restore_state_models, validate_state_invariants


async def execute_node(
    state: dict[str, Any],
    *,
    stage: str,
    node: Callable[[dict[str, Any]], Any],
) -> dict[str, Any]:
    """执行一个顶层节点，并将普通异常转换为终止状态。

    取消和键盘中断必须继续向上传播，不能被当成业务失败吞掉；其余异常
    统一生成 ``RunError``，返回最小失败状态，由图中的最终渲染节点负责
    结束本次运行。
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
        error = RunError.from_exception(stage, exc)
        # 失败事件进入状态供最终运行记录使用；生命周期日志由
        # observability.instrumentation 单独负责，避免职责重复。
        event = make_node_event(
            stage,
            "failed",
            error=error.message,
            payload={"code": error.code, "retryable": error.retryable},
        )
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
            "run": RunLifecycle(
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
            "node_events": [event],
        }
        # 边界的产物与节点产物过同一条不变量校验。此处校验不通过说明
        # 错误路径本身被改坏，是边界的 bug——向上抛出，绝不静默降级，
        # 否则“能兜住一切异常”的假象会掩盖唯一不能出错的那条路径。
        validate_state_invariants(state, failure)
        return failure
