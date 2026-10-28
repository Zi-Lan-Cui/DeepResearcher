"""串行工具的批次守卫：批内含串行工具时整批驳回，模型重新决策。"""

from typing import Any

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage


class SerialToolMiddleware(AgentMiddleware):
    """串行工具必须独占一批：违规批次在模型输出后整批驳回，不执行任何工具。"""

    def __init__(self, serial_tools: set[str]):
        super().__init__()
        self.serial_tools = frozenset(serial_tools)

    @hook_config(can_jump_to=["model"])
    async def aafter_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        messages = state.get("messages", []) if isinstance(state, dict) else []
        last = messages[-1] if messages else None
        # 无 id 的发起消息无法定点撤销，允许通过比留下半撤销状态更安全(合成测试消息才会如此)。
        if not isinstance(last, AIMessage) or not last.tool_calls or last.id is None:
            return None
        calls = list(last.tool_calls)
        offenders = sorted({call["name"] for call in calls if call["name"] in self.serial_tools})
        if not offenders or len(calls) <= 1:
            return None
        content = (
            f"本次输出包含 {len(calls)} 个工具调用，其中 {', '.join(offenders)} 是串行工具，"
            "必须单独一轮调用；整批已驳回，没有任何工具执行。"
            "请重新规划：本轮只发出一个工具调用。"
        )
        return {
            "messages": [RemoveMessage(id=last.id), HumanMessage(content=content)],
            "jump_to": "model",
        }
