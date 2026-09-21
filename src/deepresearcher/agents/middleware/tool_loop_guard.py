"""强制工具提交守卫：模型在未完成提交时输出纯文本，踢回重试。

部分 Agent（如 Writer）以工具调用作为唯一合法提交点；模型偶尔违反协议，
把本应作为工具参数提交的内容直接写进回复正文。本中间件在 after_model
观察到最后一条消息无 tool_calls 且提交探测仍未完成时，注入一条纠错
HumanMessage 并 jump 回模型，最多 max_nudges 次；耗尽后放行，
让位于业务层的兜底（如 Writer 的内联草稿救回），不制造死循环。
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
    """本回合已踢回次数走私有 state 通道。

    不放实例属性：中间件实例随编译图共享，跨运行互相污染；也不从消息历史
    推导：Summarization 压缩会整表重建历史，基于历史的计数会被静默清零、
    突破 max_nudges。UntrackedValue 与 run_model_call_count 同款——节点
    重放时预算重新起算，语义一致。
    """

    nudge_count: NotRequired[Annotated[int, UntrackedValue, PrivateStateAttr]]


class SubmittedExitMiddleware(AgentMiddleware):
    """提交点落盘后,在回模型的下一跳静默出环——不再消费任何模型调用。

    不用工具返回 Command(goto=...) 做循环控制:实测那会跳过中间件流水线
    (天花板计数、观测钩子全部空转),反复拒绝时一路撞 recursion wall。
    jump_to 是 langchain 环内唯一走完整流水线的出口通道;与 ToolLoopGuard
    同族——一个把未提交踢回模型,一个把已提交放出循环。
    """

    def __init__(self, exit_probe: Callable[[Any], bool]):
        super().__init__()
        self._exit_probe = exit_probe

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        if self._exit_probe(getattr(runtime, "context", None)):
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
            # ModelRetry 耗尽后软化的引导文本:这是后端故障,不是协议违规。
            # 踢回会让每次故障膨胀成一整轮新的重试,交给业务层兜底收敛。
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
