"""Agent 模型调用的统一重试策略。"""

import asyncio
from collections.abc import Callable
from typing import cast

from langchain.agents.middleware import ModelRetryMiddleware, ToolRetryMiddleware
from langchain_core.tools import BaseTool

from deepresearcher.llm import LLMConfigurationError, classify_llm_error
from deepresearcher.observability.usage_runtime import UsageBudgetExceeded


def retry_on(error: Exception) -> bool:
    """只重试可恢复的模型调用异常。

    配置错误、预算耗尽与账户级不可用(key/余额/硬限流)是 fail-fast 语义:
    必须冒泡到 node_runner / executor 的既有收口。在此软化它们,只会让
    agent 带着已耗尽的额度继续空烧请求,并把 terminal_reason 伪装成研究语义。
    """
    if isinstance(error, (asyncio.CancelledError, LLMConfigurationError, UsageBudgetExceeded)):
        return False
    if classify_llm_error(error) is not None:
        return False
    return True


# 软化文本的指纹:ToolLoopGuard 据此区分"模型后端故障的引导"与"协议违规的
# 纯文本"——前者踢回只会把一次故障放大成一整轮新的重试。
MODEL_FAILURE_MARKER = "的模型调用在重试后仍未成功"


def _failure_message(agent: str) -> Callable[[Exception], str]:
    def format_failure(error: Exception) -> str:
        return (
            f"{agent} {MODEL_FAILURE_MARKER}：{type(error).__name__}。"
            "请基于当前上下文调整下一步行动；不要重复提交相同的无效调用。"
        )

    return format_failure


def tool_retry_on(error: Exception) -> bool:
    """只重试工具明确标记为可恢复的错误。"""
    if isinstance(error, asyncio.CancelledError):
        return False
    return bool(getattr(error, "retryable", False)) or isinstance(
        error, (TimeoutError, ConnectionError)
    )


def _tool_failure_message(tool_label: str) -> Callable[[Exception], str]:
    def format_failure(error: Exception) -> str:
        return (
            f"{tool_label} 暂时不可用，已重试仍失败：{type(error).__name__}。"
            "请不要把该失败当作来源内容；可以更换检索式或候选来源。"
        )

    return format_failure


class _NamedToolRetryMiddleware(ToolRetryMiddleware):
    """给同一 Agent 中的多个 ToolRetryMiddleware 提供唯一名称。"""

    def __init__(
        self,
        middleware_name: str,
        *,
        tools: list[BaseTool | str],
        max_retries: int,
        retry_on: Callable[[Exception], bool],
        on_failure: Callable[[Exception], str],
        backoff_factor: float,
        initial_delay: float,
        max_delay: float,
    ) -> None:
        super().__init__(
            tools=tools,
            max_retries=max_retries,
            retry_on=retry_on,
            on_failure=on_failure,
            backoff_factor=backoff_factor,
            initial_delay=initial_delay,
            max_delay=max_delay,
        )
        self._middleware_name = middleware_name

    @property
    def name(self) -> str:
        return self._middleware_name


def model_retry(
    agent: str,
    *,
    max_retries: int = 2,
    backoff_factor: float = 2.0,
    initial_delay: float = 1.0,
    max_delay: float = 20.0,
) -> ModelRetryMiddleware:
    """创建带有项目统一错误提示的模型重试中间件。"""
    return ModelRetryMiddleware(
        max_retries=max_retries,
        retry_on=retry_on,
        on_failure=_failure_message(agent),
        backoff_factor=backoff_factor,
        initial_delay=initial_delay,
        max_delay=max_delay,
    )


def tool_retry(
    tool_names: list[str],
    tool_label: str,
    *,
    max_retries: int = 2,
    backoff_factor: float = 2.0,
    initial_delay: float = 1.0,
    max_delay: float = 20.0,
) -> ToolRetryMiddleware:
    """创建只作用于指定外部工具的重试中间件。"""
    return _NamedToolRetryMiddleware(
        middleware_name=f"ToolRetry[{tool_label}]",
        tools=cast(list[BaseTool | str], tool_names),
        max_retries=max_retries,
        retry_on=tool_retry_on,
        on_failure=_tool_failure_message(tool_label),
        backoff_factor=backoff_factor,
        initial_delay=initial_delay,
        max_delay=max_delay,
    )
