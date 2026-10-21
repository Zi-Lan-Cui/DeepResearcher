"""强制工具提交守卫：模型未完成提交就输出纯文本时，退回要求重试。

在 after_model 发现最后一条消息无 tool_calls 且提交探测未完成时，注入纠错消息并
jump 回模型，最多 max_nudges 次；耗尽后允许通过，由业务层回退。
"""

from collections.abc import Callable
from typing import Annotated, Any

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain.agents.middleware.types import AgentState, PrivateStateAttr
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.channels.untracked_value import UntrackedValue
from typing_extensions import NotRequired

from deepresearcher.agents.middleware.retry import MODEL_FAILURE_MARKER
from deepresearcher.observability.logging_config import get_logger


class ToolLoopGuardState(AgentState[Any]):
    """本回合已退回次数走私有 state 通道。

    不放实例属性（中间件实例随编译图跨运行共享），也不从消息历史推导（摘要压缩会重建历史）。
    """

    nudge_count: NotRequired[Annotated[int, UntrackedValue, PrivateStateAttr]]


class SubmittedExitMiddleware(AgentMiddleware):
    """提交点写入磁盘后，在回模型的下一跳直接退出循环，不再消费模型调用。

    用 jump_to 而非 Command(goto=...)，后者会跳过中间件流水线。probe 收 context 与
    state 双参，各 agent 只许读单一事实源，不另设平行标记。
    """

    def __init__(self, exit_probe: Callable[[Any, Any], bool]):
        super().__init__()
        self._exit_probe = exit_probe

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        if self._exit_probe(getattr(runtime, "context", None), state):
            return {"jump_to": "end"}
        return None


class ToolLoopGuardMiddleware(AgentMiddleware):
    """“文本不算提交”的循环内守卫；submitted_probe 返回 True 后不再拦截。"""

    state_schema = ToolLoopGuardState  # type: ignore[assignment]

    def __init__(
        self,
        agent_name: str,
        nudge_message: str,
        submitted_probe: Callable[[Any], bool],
        max_nudges: int = 2,
        run_limit: int = 0,
        reminder_message: str = "",
        reminder_turns: int = 0,
        emit: Callable[[str, dict[str, object]], None] | None = None,
        event_slug: str | None = None,
    ):
        super().__init__()
        self.agent_name = agent_name
        self._slug = event_slug or agent_name.lower()
        self.nudge_message = nudge_message
        self.submitted_probe = submitted_probe
        self.max_nudges = max_nudges
        self.run_limit = run_limit
        self.reminder_message = reminder_message
        self.reminder_turns = reminder_turns
        self._emit = emit
        self._logger = get_logger("deepresearcher.agents.middleware.tool_loop_guard")

    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """最后几回合显式告知剩余预算，促使模型先提交已验证材料。"""
        if not self.reminder_message or self.reminder_turns <= 0 or self.run_limit <= 0:
            return None
        if self.submitted_probe(getattr(runtime, "context", None)):
            return None
        messages = state.get("messages", []) if isinstance(state, dict) else []
        calls = int(state.get("run_model_call_count", 0) or 0) if isinstance(state, dict) else 0
        remaining = self.run_limit - calls
        if remaining <= 0 or remaining > self.reminder_turns:
            return None
        content = self.reminder_message.format(remaining_turns=remaining)
        if any(
            isinstance(message, HumanMessage) and message.content == content for message in messages
        ):
            return None
        payload = {
            "agent": self.agent_name,
            "remaining_turns": remaining,
            "run_limit": self.run_limit,
        }
        if self._emit is not None:
            self._emit(f"{self._slug}_finalization_reminded", payload)
        else:
            self._logger.info("%s_finalization_reminded payload=%s", self._slug, payload)
        return {"messages": [HumanMessage(content=content)]}

    @hook_config(can_jump_to=["model"])
    async def aafter_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        messages = state.get("messages", []) if isinstance(state, dict) else []
        last = messages[-1] if messages else None
        if not isinstance(last, AIMessage) or last.tool_calls:
            return None
        if MODEL_FAILURE_MARKER in (last.text or ""):
            # ModelRetry 耗尽后输出的引导文本：这是后端故障，不是协议违规；
            # 退回会把一次故障放大成一整轮新的重试，交由业务层处理。
            return None
        if self.submitted_probe(getattr(runtime, "context", None)):
            return None
        nudges = int(state.get("nudge_count", 0) or 0) if isinstance(state, dict) else 0
        if nudges >= self.max_nudges:
            return None
        payload = {
            "agent": self.agent_name,
            "nudge": nudges + 1,
            "max_nudges": self.max_nudges,
            "content_chars": len(last.text or ""),
        }
        if self._emit is not None:
            self._emit(f"{self._slug}_tool_loop_nudged", payload)
        else:
            self._logger.info("%s_tool_loop_nudged payload=%s", self._slug, payload)
        return {
            "messages": [HumanMessage(content=self.nudge_message)],
            "jump_to": "model",
            "nudge_count": nudges + 1,
        }
