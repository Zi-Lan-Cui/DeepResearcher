"""共享运行状态工具的串行执行控制。"""

from typing import Any, cast

from langchain.agents.middleware import AgentMiddleware


class SerialToolMiddleware(AgentMiddleware):
    """只串行化共享本次运行状态的工具，不阻塞研究任务并发。"""

    def __init__(self, serial_tools: set[str]):
        super().__init__()
        self.serial_tools = serial_tools

    async def awrap_tool_call(self, request, handler):
        if request.tool_call["name"] in self.serial_tools:
            context = cast(Any, request.runtime.context)
            async with context.tool_lock:
                return await handler(request)
        return await handler(request)
