"""Agent 共享机器件之装配:中间件栈的声明(Profile)与组装(Builder)同文件。

Profile 收拢 ``build_agent_middleware`` 的长参数列表:每个 Agent 用一份 Profile
描述自己要的中间件栈,参数间的成对约束(如提交守卫的消息与探测)由结构本身
保证,而不是靠两个可空参数之间的隐式约定——声明与唯一装配者住在一起,读栈
的形状只需要这一个文件。
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
from deepresearcher.tokens import get_token_estimator

_TOKEN_ESTIMATOR = get_token_estimator()
AGENT_RECURSION_LIMIT = 1_000


class ObservableSummarizationMiddleware(SummarizationMiddleware):
    """压缩发生时记一笔 context_compacted——纯观测,不改任何压缩决策。

    这组数据是将来裁定 clear_tool_inputs(是否连工具入参一起清)的唯一
    合法依据:先看真实长 run 触发几次、清完还剩多少,再谈调参。
    """

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
                    "context_compacted",
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
    """工具提交守卫配置：成对出现，杜绝只配一半。

    模型在完成提交（``submitted_probe`` 返回 False）前输出纯文本时，
    注入 ``nudge_message`` 并踢回重试；耗尽后放行给业务层兜底。
    """

    nudge_message: str
    submitted_probe: Callable[[Any], bool]
    max_nudges: int = 2
    reminder_message: str = ""
    reminder_turns: int = 0


@dataclass(frozen=True)
class MiddlewareProfile:
    """一个 Agent 的中间件栈完整输入。

    ``max_turns`` 只是防失控的模型调用天花板，不承担业务配额语义；
    业务配额由 ``tool_call_limits``（单工具调用数）与业务层 hard check 表达。
    """

    agent_name: str
    event_slug: str  # 事件名前缀(snake 归因域,如 "researcher");与显示名 agent_name 分家
    max_turns: int
    context_window_tokens: int
    model: BaseChatModel | None = None
    retry_tools: Sequence[list[str]] = ()
    serial_tools: set[str] | None = None
    tool_call_limits: Sequence[tuple[str, int]] = ()
    submission_guard: SubmissionGuard | None = None
    # 提交落盘后由 SubmittedExitMiddleware 在下一跳静默出环(Supervisor 的
    # ResearchComplete 用它终结;工具本身不设 return_direct)。None = 不挂。
    exit_probe: Callable[[Any], bool] | None = None
    emit: AgentEmit | None = None  # 契约见 observability.events.AgentEmit


def _message_text(message: BaseMessage) -> str:
    """把消息中用于计数的内容规范化为字符串(含工具调用参数)。

    tool_calls 的参数是真实载荷——Writer 草稿、AddEvidence 逐字引用都住在这里;
    只数 content 会让两级压缩闸对最重的消息失明。
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

    Profile 字段语义见本文件 ``MiddlewareProfile`` 的类/字段注释；注册顺序即行为契约：提交守卫
    先于 AgentObservability 注册（after-hook 逆序执行），保证回合日志完整记录
    被拦截的文本输出；ModelCallLimit 最后注册，其计数先于日志中间件递增。
    """
    trigger = max(1_024, int(profile.context_window_tokens * 0.8))
    keep = max(1_024, int(profile.context_window_tokens * 0.25))
    middleware: list[AgentMiddleware] = [
        ContextEditingMiddleware(
            edits=[ClearToolUsesEdit(trigger=trigger, keep=3)],
            token_counter=count_message_tokens,
        )
    ]
    # 压缩中间件会向模型要真实 BaseChatModel 能力(with_retry 等)；None 与测试
    # 窄假模型都按"未配置压缩模型"处理——生产恒为 ChatOpenAI,始终启用压缩。
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
    # 标签默认取工具名本身(retry_tools 只声明工具;展示名即协议名)。
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
