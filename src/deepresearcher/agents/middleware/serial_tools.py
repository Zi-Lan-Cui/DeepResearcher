"""共享运行状态工具的串行执行控制。"""

from typing import Any, cast

from langchain.agents.middleware import AgentMiddleware


class SerialToolMiddleware(AgentMiddleware):
    """将状态边界工具作为独占屏障，其余工具允许共享并发。"""

    def __init__(self, serial_tools: set[str]):
        super().__init__()
        self.serial_tools = serial_tools

    async def awrap_tool_call(self, request, handler):
        context = cast(Any, request.runtime.context)
        gate = getattr(context, "tool_gate", None)
        if gate is not None:
            guard = (
                gate.exclusive()
                if request.tool_call["name"] in self.serial_tools
                else gate.shared()
            )
            async with guard:
                return await handler(request)
        if request.tool_call["name"] in self.serial_tools:
            async with context.tool_lock:
                return await handler(request)
        return await handler(request)
