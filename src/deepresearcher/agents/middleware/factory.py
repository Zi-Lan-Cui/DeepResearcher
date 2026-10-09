"""Agent 共享中间件的装配:栈声明 MiddlewareProfile 与装配函数 build_agent_middleware 同文件。

Profile 收拢 build_agent_middleware 的长参数列表,每个 Agent 用一份 Profile 描述自己要的
中间件栈;提交守卫的消息与探测这类成对约束由结构本身保证,不靠两个可空参数之间的隐式约定。
"""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

from langchain.agents.middleware import (
    AgentMiddleware,
    ClearToolUsesEdit,
    ContextEditingMiddleware,
    ModelCallLimitMiddleware,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from deepresearcher.agents.middleware.observability import AgentObservabilityMiddleware
from deepresearcher.agents.middleware.retry import (
    ToolErrorNormalizerMiddleware,
    model_retry,
    tool_retry,
)
from deepresearcher.agents.middleware.serial_tools import SerialToolMiddleware
from deepresearcher.agents.middleware.tool_loop_guard import (
    SubmittedExitMiddleware,
    ToolLoopGuardMiddleware,
)
from deepresearcher.observability.events import AgentEmit
from deepresearcher.observability.events.names import EventName
from deepresearcher.tokens import get_token_estimator

_TOKEN_ESTIMATOR = get_token_estimator()
AGENT_RECURSION_LIMIT = 1_000


class ObservableSummarizationMiddleware(SummarizationMiddleware):
    """压缩发生时记一笔 context_compacted;只记录,不参与任何压缩决策。"""

    def __init__(self, *, agent_name: str, emit: Any, trigger_tokens: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._agent_name = agent_name
        self._emit = emit
        self._trigger_tokens = trigger_tokens

    async def abefore_model(self, state: Any, runtime: Any) -> Any:
        update = await super().abefore_model(state, runtime)
        messages = state.get("messages", []) if isinstance(state, dict) else []
        if update and self._emit is not None:
            removed = next(
                (
                    message
                    for message in update.get("messages", [])
                    if isinstance(message, RemoveMessage) and message.id == REMOVE_ALL_MESSAGES
                ),
                None,
            )
            if removed is not None:
                kept = [message for message in update["messages"] if message is not removed]
                self._emit(
                    EventName.CONTEXT_COMPACTED,
                    {
                        "agent": self._agent_name,
                        "before_tokens": count_message_tokens(messages),
                        "after_tokens": count_message_tokens(kept),
                        "trigger_tokens": self._trigger_tokens,
                    },
                )
        return update


@dataclass(frozen=True)
class SubmissionGuard:
    """工具提交守卫配置;消息与探测绑定在同一数据类里,不会只配一半。

    模型在 ``submitted_probe`` 返回 False 前输出纯文本时,注入
    ``nudge_message`` 并退回重试;退回次数耗尽后放行,交业务层兜底。
    """

    nudge_message: str
    submitted_probe: Callable[[Any], bool]
    max_nudges: int = 2
    reminder_message: str = ""
    reminder_turns: int = 0


@dataclass(frozen=True)
class MiddlewareProfile:
    """一个 Agent 的中间件栈完整输入。

    ``max_turns`` 只是防失控的模型调用次数上限,不承担业务配额语义;业务配额由
    ``tool_call_limits`` 的单工具调用数与业务层硬校验表达。
    """

    agent_name: str
    event_slug: str  # 事件名前缀,snake_case 风格,如 researcher;与显示名 agent_name 分开设置
    max_turns: int
    context_window_tokens: int
    model: BaseChatModel | None = None
    retry_tools: Sequence[list[str]] = ()
    serial_tools: set[str] | None = (
        None  # 独批工具:由 SerialToolMiddleware 在模型输出后校验,混批即整批驳回、打回模型重发
    )
    tool_call_limits: Sequence[tuple[str, int]] = ()
    submission_guard: SubmissionGuard | None = None
    # 提交落盘后由 SubmittedExitMiddleware 在下一跳直接退出循环;probe 收 context 与 state,
    # 读该 agent 提交事实所在的单一事实源;工具一律不设 return_direct、不用 Command(goto)
    # 控环,被拒后模型必须留在循环内改正。None 表示不挂。
    exit_probe: Callable[[Any, Any], bool] | None = None
    emit: AgentEmit | None = None  # 契约见 observability.events.AgentEmit


def _message_text(message: BaseMessage) -> str:
    """把消息中用于计数的内容规范化为字符串,含工具调用参数。

    tool_calls 的参数也是真实载荷:Writer 草稿、AddEvidence 逐字引用都在其中,
    只数 content 会让两级压缩对最重的消息不生效。
    """
    parts = [message.content if isinstance(message.content, str) else str(message.content)]
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        parts.append(json.dumps(tool_calls, ensure_ascii=False, default=str))
    return "\n".join(parts)


def count_message_tokens(messages) -> int:
    """使用项目 tokenizer 估算 LangChain 消息总量。"""
    return sum(_TOKEN_ESTIMATOR.count(_message_text(message)) for message in messages)


def build_agent_middleware(profile: MiddlewareProfile) -> list[AgentMiddleware]:
    """按 Profile 组装所有 Agent 共用的模型、工具、上下文和轮次中间件。

    Profile 字段语义见本文件 ``MiddlewareProfile`` 的类与字段注释。注册顺序即行为契约:
    提交守卫先于 AgentObservability 注册,after-hook 逆序执行,回合日志因此能完整记录
    被拦截的文本输出;ModelCallLimit 最后注册,其计数先于日志中间件递增。
    """
    trigger = max(1_024, int(profile.context_window_tokens * 0.8))
    keep = max(1_024, int(profile.context_window_tokens * 0.25))
    middleware: list[AgentMiddleware] = [
        ContextEditingMiddleware(
            edits=[ClearToolUsesEdit(trigger=trigger, keep=3)],
            token_counter=count_message_tokens,
        )
    ]
    # 压缩中间件要求模型具备真实 BaseChatModel 能力,如 with_retry;None 与测试用的窄假
    # 模型都按"未配置压缩模型"处理。生产 profile.model 恒为 ChatOpenAI,压缩始终启用。
    if isinstance(profile.model, BaseChatModel):
        middleware.insert(
            0,
            ObservableSummarizationMiddleware(
                agent_name=profile.agent_name,
                emit=profile.emit,
                trigger_tokens=trigger,
                model=profile.model,
                trigger=("tokens", trigger),
                keep=("tokens", keep),
                token_counter=count_message_tokens,
            ),
        )
    middleware.append(
        model_retry(profile.agent_name, event_slug=profile.event_slug, emit=profile.emit)
    )
    # 标签取工具名本身:retry_tools 只声明工具,展示名即协议名。
    middleware.extend(tool_retry(names, names[0]) for names in profile.retry_tools)
    if profile.retry_tools:
        # 归一层必须注册在 tool_retry 之后(更内):retry 看到的是归一后的异常,
        # observability(更更内)记录的仍是原始异常。
        middleware.append(ToolErrorNormalizerMiddleware())
    if profile.serial_tools:
        middleware.append(SerialToolMiddleware(profile.serial_tools))
    if profile.exit_probe is not None:
        middleware.append(SubmittedExitMiddleware(profile.exit_probe))
    for tool_name, tool_call_limit in profile.tool_call_limits:
        middleware.append(
            cast(
                AgentMiddleware,
                ToolCallLimitMiddleware(
                    tool_name=tool_name, run_limit=tool_call_limit, exit_behavior="continue"
                ),
            )
        )
    if profile.submission_guard is not None:
        middleware.append(
            ToolLoopGuardMiddleware(
                agent_name=profile.agent_name,
                event_slug=profile.event_slug,
                nudge_message=profile.submission_guard.nudge_message,
                submitted_probe=profile.submission_guard.submitted_probe,
                max_nudges=profile.submission_guard.max_nudges,
                run_limit=profile.max_turns,
                reminder_message=profile.submission_guard.reminder_message,
                reminder_turns=profile.submission_guard.reminder_turns,
                emit=profile.emit,
            )
        )
    middleware.append(
        AgentObservabilityMiddleware(
            agent_name=profile.agent_name,
            run_limit=profile.max_turns,
            emit=profile.emit,
            event_slug=profile.event_slug,
        )
    )
    middleware.append(
        cast(
            AgentMiddleware,
            ModelCallLimitMiddleware(run_limit=profile.max_turns, exit_behavior="end"),
        )
    )
    return middleware
