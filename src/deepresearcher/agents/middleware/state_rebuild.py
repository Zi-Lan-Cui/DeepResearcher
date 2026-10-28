"""内层循环每模型回合前从通道重建业务状态对象的中间件。"""

from typing import Any

from langchain.agents.middleware import AgentMiddleware


class StateRebuildMiddleware(AgentMiddleware):
    """每模型回合前从通道重建本轮的状态对象。"""

    async def abefore_model(self, state: Any, runtime: Any) -> None:
        context = getattr(runtime, "context", None)
        rebuild = getattr(context, "rebuild_state", None)
        if callable(rebuild):
            rebuild(state)
        return
