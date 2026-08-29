"""Agent 每轮模型调用的可观测性中间件。

记录每次模型回合实际调用的工具、回合数与最终终止原因，使“达到
ModelCallLimit 后未提交”这类失败可以从日志中直接读出每轮的动作，
而不需要推断。事件与 Agent 自身的审计事件走同一 sink。
"""

from collections.abc import Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage

from deepsearch_agent.observability.logger import get_logger

_LIMIT_MESSAGE_MARKER = "Model call limits exceeded"
# ModelCallLimitMiddleware(exit_behavior="end") 注入的收尾消息以此开头；
# Agent 业务层可用它区分“被预算掐断”与“模型自行收尾”。
LIMIT_MESSAGE_MARKER = _LIMIT_MESSAGE_MARKER
# supervisor 的回合文字会经 plan 帧直接上屏，并与逐字流式同源：
# 预览若比流式短，聚合替换瞬间会出现肉眼可见的"打字完被截断"。
# 800 是单回合旁白的展示上限（事件体积仍受 observability 有界化约束）。
_PREVIEW_CHARS = 800


class TurnLoggingMiddleware(AgentMiddleware):
    """每次模型输出后记录 tool_name/turn，Agent 结束时记录终止原因。

    计数一律取自 langgraph 的每次运行 state（run_model_call_count），
    不存在实例属性上：中间件实例随编译图共享，并发 ainvoke 会互相污染。
    """

    def __init__(
        self,
        agent_name: str,
        run_limit: int,
        emit: Callable[[str, dict[str, object]], None] | None = None,
    ):
        super().__init__()
        self.agent_name = agent_name
        self.run_limit = run_limit
        self._emit = emit
        self._logger = get_logger("deepsearch_agent.agents.middleware.turn_logging")

    @staticmethod
    def _call_count(state: Any) -> int:
        if not isinstance(state, dict):
            return 0
        return int(state.get("run_model_call_count", 0) or 0)

    async def aafter_model(self, state: Any, runtime: Any) -> None:
        messages = state.get("messages", []) if isinstance(state, dict) else []
        last = messages[-1] if messages else None
        if not isinstance(last, AIMessage):
            return
        turn = self._call_count(state)
        tool_names = [str(call.get("name", "")) for call in last.tool_calls or []]
        content = last.text or ""
        self._log_event(
            f"{self.agent_name.lower()}_model_turn",
            {
                "agent": self.agent_name,
                "turn": turn,
                "model_call_count": turn,
                "run_limit": self.run_limit,
                "tool_names": tool_names,
                "content_chars": len(content),
                "content_preview": content[:_PREVIEW_CHARS],
            },
        )

    async def aafter_agent(self, state: Any, runtime: Any) -> None:
        messages = state.get("messages", []) if isinstance(state, dict) else []
        last = messages[-1] if messages else None
        reason = "final_response"
        if isinstance(last, AIMessage):
            content = last.text or ""
            if _LIMIT_MESSAGE_MARKER in content:
                reason = "model_call_limit_exceeded"
            elif last.tool_calls:
                reason = "ended_on_tool_call_turn"
        self._log_event(
            f"{self.agent_name.lower()}_agent_finished",
            {
                "agent": self.agent_name,
                "turns": self._call_count(state),
                "run_limit": self.run_limit,
                "stop_reason": reason,
            },
        )

    def _log_event(self, event_type: str, payload: dict[str, object]) -> None:
        if self._emit is not None:
            self._emit(event_type, payload)
            return
        self._logger.info("%s payload=%s", event_type, payload)
