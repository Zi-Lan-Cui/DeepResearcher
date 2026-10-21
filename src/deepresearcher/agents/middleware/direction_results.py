"""每模型回合前把已完成的方向结果注入上下文的中间件。

必须排在 LoopStateRebuildMiddleware 之后：先重建 loop_state 再接收，否则重建会用
通道旧值覆盖本钩子的结果。context 提供 direction_tasks 与 loop_state 即可挂载。
"""

from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage

from deepresearcher.prompts import render_data_section


class DirectionResultReceiverMiddleware(AgentMiddleware):
    """每模型回合前接收已完成方向；无结果与无待清算时返回 None。"""

    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        from deepresearcher.agents.supervisor import services

        context = getattr(runtime, "context", None)
        registry = getattr(context, "direction_tasks", None)
        loop_state = getattr(context, "loop_state", None)
        if context is None or registry is None or loop_state is None:
            return None
        if not registry and not loop_state.pending_directions:
            return None

        reports, delta = services.receive_ready_directions(
            context.deps, loop_state, registry, merged_via="model_turn"
        )
        if not reports:
            return None
        payload: dict[str, object] = {
            "directions": reports,
            "workbench_revision": loop_state.working_set_revision,
            "inflight": [
                {
                    "task_id": key,
                    "display_title": str((item or {}).get("display_title") or ""),
                }
                for key, item in sorted(loop_state.pending_directions.items())
            ],
        }
        message = HumanMessage(content=render_data_section("研究管理观察（方向结果）", payload))
        return {"messages": [message], **delta}
