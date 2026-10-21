"""内层循环每模型回合前从通道重建业务状态对象的中间件。

每回合前调 context.rebuild_loop_state(state)，用通道现值替换上一批的内存对象，
未提交的改动因此被丢弃。凡 context 提供 rebuild_loop_state 即可挂载。
"""

from typing import Any

from langchain.agents.middleware import AgentMiddleware


class LoopStateRebuildMiddleware(AgentMiddleware):
    """每模型回合前从通道重建 loop_state。"""

    async def abefore_model(self, state: Any, runtime: Any) -> None:
        context = getattr(runtime, "context", None)
        rebuild = getattr(context, "rebuild_loop_state", None)
        if callable(rebuild):
            rebuild(state)
        return None
