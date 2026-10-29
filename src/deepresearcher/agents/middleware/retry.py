"""Agent 模型调用的统一重试策略。"""

from collections.abc import Callable
from typing import cast

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRetryMiddleware,
    ToolRetryMiddleware,
)
from langchain_core.tools import BaseTool

from deepresearcher.backoff import DEFAULT_BACKOFF_FACTOR
from deepresearcher.llm import LLMConfigurationError, classify_llm_error
from deepresearcher.tools.errors import ToolError
from deepresearcher.usage_runtime import UsageBudgetExceeded


def retry_on(error: Exception) -> bool:
    """只重试可恢复的模型调用异常。

    配置错误、预算耗尽与账户级不可用属 fail-fast，必须向上传播；在本层重试只会在
    额度已耗尽时继续发无效请求。CancelledError 是 BaseException，不经此处。
    """
    if isinstance(error, (LLMConfigurationError, UsageBudgetExceeded)):
        return False
    if classify_llm_error(error) is not None:
        return False
    return True


class ToolErrorNormalizerMiddleware(AgentMiddleware):
    """工具异常边界归一：重试层的判定依据是构造期声明，不是异常类型猜测。

    工具重试的唯一准入是 ToolError.retryable；未标记的意外异常在本层补声明：
    TimeoutError 与 ConnectionError 可重试，其余按 bug 判不可重试。
    """

    @property
    def name(self) -> str:
        return "ToolErrorNormalizer"

    async def awrap_tool_call(self, request, handler):  # type: ignore[override]
        try:
            return await handler(request)
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(
                f"工具内部意外错误：{type(exc).__name__}: {exc}",
                code="tool_unexpected",
                retryable=isinstance(exc, (TimeoutError, ConnectionError)),
            ) from exc


# 模型失败标记
MODEL_FAILURE_MARKER = "的模型调用在重试后仍未成功"


def _failure_message(agent: str) -> Callable[[Exception], str]:
    def format_failure(error: Exception) -> str:
        return (
            f"{agent} {MODEL_FAILURE_MARKER}：{type(error).__name__}。"
            "请基于当前上下文调整下一步行动；不要重复提交相同的无效调用。"
        )

    return format_failure


def tool_retry_on(error: Exception) -> bool:
    """只重试自标可恢复的错误；未标 ToolError 的意外异常由边界归一层补声明。"""
    return bool(getattr(error, "retryable", False))


def _tool_failure_message(tool_label: str) -> Callable[[Exception], str]:
    def format_failure(error: Exception) -> str:
        return (
            f"{tool_label} 暂时不可用，已重试仍失败：{type(error).__name__}。"
            "请不要把该失败当作来源内容；可更换调用参数或改用其他工具。"
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


_RETRY_MAX_RETRIES: int = 2
_RETRY_INITIAL_DELAY_SECONDS: float = 1.0
_RETRY_MAX_DELAY_SECONDS: float = 20.0


def model_retry(
    agent: str,
    *,
    event_slug: str | None = None,
    max_retries: int = _RETRY_MAX_RETRIES,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    initial_delay: float = _RETRY_INITIAL_DELAY_SECONDS,
    max_delay: float = _RETRY_MAX_DELAY_SECONDS,
    emit: Callable[[str, dict[str, object]], None] | None = None,
) -> ModelRetryMiddleware:
    """创建带有项目统一错误提示的模型重试中间件。

    事件只在重试耗尽时发出，此时本回合已消耗 max_retries+1 次请求。
    """

    def on_failure(error: Exception) -> str:
        if emit is not None:
            emit(
                f"{(event_slug or agent.lower())}_model_retry_exhausted",
                {"agent": agent, "error_type": type(error).__name__, "max_retries": max_retries},
            )
        return _failure_message(agent)(error)

    return ModelRetryMiddleware(
        max_retries=max_retries,
        retry_on=retry_on,
        on_failure=on_failure,
        backoff_factor=backoff_factor,
        initial_delay=initial_delay,
        max_delay=max_delay,
    )


def tool_retry(
    tool_names: list[str],
    tool_label: str,
    *,
    max_retries: int = _RETRY_MAX_RETRIES,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    initial_delay: float = _RETRY_INITIAL_DELAY_SECONDS,
    max_delay: float = _RETRY_MAX_DELAY_SECONDS,
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
