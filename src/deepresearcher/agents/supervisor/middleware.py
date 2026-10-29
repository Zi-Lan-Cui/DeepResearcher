"""Supervisor 专属中间件：每模型回合前把已完成的任务结果注入上下文。

必须排在 StateRebuildMiddleware 之后：先重建 round_state 再接收，否则重建会用
通道旧值覆盖本钩子的结果。context 提供 task_handles 与 round_state 即可挂载。
"""

from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage

from deepresearcher.agents.supervisor.services import receive_ready_tasks
from deepresearcher.prompts import render_data_section


class DirectionResultReceiverMiddleware(AgentMiddleware):
    """每模型回合前接收已完成任务；无结果与无待清算时返回 None。"""

    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        context = getattr(runtime, "context", None)
        registry = getattr(context, "task_handles", None)
        round_state = getattr(context, "round_state", None)
        if context is None or registry is None or round_state is None:
            return None
        if not registry and not round_state.pending_tasks:
            return None

        reports, delta = receive_ready_tasks(
            context.deps, round_state, registry, merged_via="model_turn"
        )
        if not reports:
            return None
        payload: dict[str, object] = {
            "tasks": reports,
            "workbench_revision": round_state.working_set_revision,
            "inflight": [
                {
                    "task_id": key,
                    "display_title": str((item or {}).get("display_title") or ""),
                }
                for key, item in sorted(round_state.pending_tasks.items())
            ],
        }
        message = HumanMessage(content=render_data_section("研究管理观察（任务结果）", payload))
        return {"messages": [message], **delta}
