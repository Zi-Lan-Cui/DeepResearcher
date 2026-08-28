"""强制工具提交守卫：模型在未完成提交时输出纯文本，踢回重试。

部分 Agent（如 Writer）以工具调用作为唯一合法提交点；模型偶尔违反协议，
把本应作为工具参数提交的内容直接写进回复正文。本中间件在 after_model
观察到最后一条消息无 tool_calls 且提交探测仍未完成时，注入一条纠错
HumanMessage 并 jump 回模型，最多 max_nudges 次；耗尽后放行，
让位于业务层的兜底（如 Writer 的内联草稿救回），不制造死循环。
"""

from collections.abc import Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, HumanMessage

from deepsearch_agent.observability.logger import get_logger


class ToolLoopGuardMiddleware(AgentMiddleware):
    """“文本不算提交”的循环内守卫；submitted_probe 返回 True 后不再拦截。

    已踢回次数从本次运行的消息历史推导（匹配注入过的提示原文），
    不放实例属性：中间件实例随编译图共享，跨运行计数会互相污染。
    """

    def __init__(
        self,
        agent_name: str,
        nudge_message: str,
        submitted_probe: Callable[[Any], bool],
        max_nudges: int = 2,
        emit: Callable[[str, dict[str, object]], None] | None = None,
    ):
        super().__init__()
        self.agent_name = agent_name
        self.nudge_message = nudge_message
        self.submitted_probe = submitted_probe
        self.max_nudges = max_nudges
        self._emit = emit
        self._logger = get_logger("deepsearch_agent.agents.middleware.tool_loop_guard")

    @hook_config(can_jump_to=["model"])
    async def aafter_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        messages = state.get("messages", []) if isinstance(state, dict) else []
        last = messages[-1] if messages else None
        if not isinstance(last, AIMessage) or last.tool_calls:
            return None
        if self.submitted_probe(getattr(runtime, "context", None)):
            return None
        nudges = sum(
            1
            for message in messages
            if isinstance(message, HumanMessage) and message.content == self.nudge_message
        )
        if nudges >= self.max_nudges:
            return None
        payload = {
            "agent": self.agent_name,
            "nudge": nudges + 1,
            "max_nudges": self.max_nudges,
            "content_chars": len(last.text or ""),
        }
        if self._emit is not None:
            self._emit(f"{self.agent_name.lower()}_tool_loop_nudged", payload)
        else:
            self._logger.info("%s_tool_loop_nudged payload=%s", self.agent_name.lower(), payload)
        return {"messages": [HumanMessage(content=self.nudge_message)], "jump_to": "model"}
