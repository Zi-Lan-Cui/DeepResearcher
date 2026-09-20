"""Agent 中间件的统一装配入口。"""

from typing import cast

from langchain.agents.middleware import (
    AgentMiddleware,
    ClearToolUsesEdit,
    ContextEditingMiddleware,
    ModelCallLimitMiddleware,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

from deepresearcher.agents.middleware.observability import AgentObservabilityMiddleware
from deepresearcher.agents.middleware.profile import MiddlewareProfile
from deepresearcher.agents.middleware.retry import model_retry, tool_retry
from deepresearcher.evidence.tokens import get_token_estimator

_TOKEN_ESTIMATOR = get_token_estimator()
AGENT_RECURSION_LIMIT = 1_000


def _message_text(message: BaseMessage) -> str:
    """把消息中用于计数的内容规范化为字符串。"""
    content = message.content
    return content if isinstance(content, str) else str(content)


def count_message_tokens(messages) -> int:
    """使用项目 tokenizer 估算 LangChain 消息总量。"""
    return sum(_TOKEN_ESTIMATOR.count(_message_text(message)) for message in messages)


def build_agent_middleware(profile: MiddlewareProfile) -> list[AgentMiddleware]:
    """按 Profile 组装所有 Agent 共用的模型、工具、上下文和轮次中间件。

    Profile 语义约定见 :mod:`.profile`；注册顺序即行为契约：提交守卫
    先于 AgentObservability 注册（after-hook 逆序执行），保证回合日志完整记录
    被拦截的文本输出；ModelCallLimit 最后注册，其计数先于日志中间件递增。
    """
    from deepresearcher.agents.middleware.serial_tools import SerialToolMiddleware
    from deepresearcher.agents.middleware.tool_loop_guard import ToolLoopGuardMiddleware

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
            SummarizationMiddleware(
                model=profile.model,
                trigger=("tokens", trigger),
                keep=("tokens", keep),
                token_counter=count_message_tokens,
            ),
        )
    middleware.append(model_retry(profile.agent_name))
    middleware.extend(tool_retry(names, label) for names, label in profile.retry_tools)
    if profile.serial_tools:
        middleware.append(SerialToolMiddleware(profile.serial_tools))
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
            agent_name=profile.agent_name, run_limit=profile.max_turns, emit=profile.emit
        )
    )
    middleware.append(
        cast(
            AgentMiddleware,
            ModelCallLimitMiddleware(run_limit=profile.max_turns, exit_behavior="end"),
        )
    )
    return middleware
