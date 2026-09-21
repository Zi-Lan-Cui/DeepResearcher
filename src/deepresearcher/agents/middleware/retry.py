"""Agent 模型调用的统一重试策略。"""

from collections.abc import Callable
from typing import cast

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRetryMiddleware,
    ToolRetryMiddleware,
)
from langchain_core.tools import BaseTool

from deepresearcher.llm import LLMConfigurationError, classify_llm_error
from deepresearcher.observability.usage_runtime import UsageBudgetExceeded
from deepresearcher.tools.errors import ToolError


def retry_on(error: Exception) -> bool:
    """只重试可恢复的模型调用异常。

    配置错误、预算耗尽与账户级不可用(key/余额/硬限流)是 fail-fast 语义:
    必须冒泡到 node_runner / executor 的既有收口。在此软化它们,只会让
    agent 带着已耗尽的额度继续空烧请求,并把 terminal_reason 伪装成研究语义。

    取消不经此处:CancelledError 是 BaseException,库的重试循环只捕 Exception。
    """
    if isinstance(error, (LLMConfigurationError, UsageBudgetExceeded)):
        return False
    if classify_llm_error(error) is not None:
        return False
    return True


class ToolErrorNormalizerMiddleware(AgentMiddleware):
    """工具异常边界归一:重试层的判据是构造期声明,不是异常类型猜测。

    工具重试的唯一准入是 ToolError.retryable;未被标记的意外异常在边界上
    按性质声明:传输类(内置 TimeoutError/ConnectionError——direct fetch 就
    显式抛这两个)可重试,其余 bug 级判不可重试。观测中间件注册在本层之内,
    记录到的仍是原始异常;归一只对重试语义生效。
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
    """只重试自标可恢复的错误;未标 ToolError 的意外异常由边界归一层补声明。"""
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
_RETRY_BACKOFF: dict[str, float] = {"backoff_factor": 2.0, "initial_delay": 1.0, "max_delay": 20.0}


def model_retry(
    agent: str,
    *,
    max_retries: int = _RETRY_MAX_RETRIES,
    backoff_factor: float = _RETRY_BACKOFF["backoff_factor"],
    initial_delay: float = _RETRY_BACKOFF["initial_delay"],
    max_delay: float = _RETRY_BACKOFF["max_delay"],
    emit: Callable[[str, dict[str, object]], None] | None = None,
) -> ModelRetryMiddleware:
    """创建带有项目统一错误提示的模型重试中间件。

    逐次尝试由库内部循环、无处挂钩;可观测的锚点是耗尽这一确定时刻——
    它意味着本回合白白烧掉 max_retries+1 次请求,事件流此前完全隐身。
    """

    def on_failure(error: Exception) -> str:
        if emit is not None:
            emit(
                f"{agent.lower()}_model_retry_exhausted",
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
    backoff_factor: float = _RETRY_BACKOFF["backoff_factor"],
    initial_delay: float = _RETRY_BACKOFF["initial_delay"],
    max_delay: float = _RETRY_BACKOFF["max_delay"],
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
