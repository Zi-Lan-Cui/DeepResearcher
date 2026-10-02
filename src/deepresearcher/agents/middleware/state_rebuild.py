"""内层循环每模型回合前从通道重建业务状态对象的中间件。

通道增量在工具批末随 superstep 落盘;批内工具共享的 loop_state 不进 checkpoint。
本中间件在每个模型回合前调 `context.rebuild_loop_state(state)`,用通道现值替换上一批的
内存对象,使未经 Command 提交的改动被丢弃、批内起点始终等于已提交状态。凡 context 提供
rebuild_loop_state 即可挂。
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
