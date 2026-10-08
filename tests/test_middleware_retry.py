"""模型与工具两层重试判据、工具异常边界归一的单元测试。"""

import asyncio

import pytest

from deepresearcher.agents.middleware.retry import (
    ToolErrorNormalizerMiddleware,
    _failure_message,
    retry_on,
    tool_retry_on,
)
from deepresearcher.llm import LLMConfigurationError
from deepresearcher.tools.errors import ToolError, ToolRequestError
from deepresearcher.usage_runtime import UsageBudgetExceeded


def test_model_retry_predicate():
    assert retry_on(ConnectionResetError("抖动")) is True  # 未分类默认可恢复
    assert retry_on(LLMConfigurationError("缺 key")) is False
    assert retry_on(UsageBudgetExceeded("budget")) is False

    class Fatal(Exception):
        status_code = 401

    assert retry_on(Fatal()) is False  # 账户级不可用 fail-fast


def test_tool_retry_predicate_trusts_only_declared_retryability():
    assert tool_retry_on(ToolRequestError("429 稍后")) is True
    assert tool_retry_on(ToolError("配置缺失")) is False
    # 内建 Timeout/Connection 不作为判据:它们与真实工具抛出的异常类型不相交,
    # 未知异常统一交给边界归一层处理。
    assert tool_retry_on(TimeoutError("永不命中的形状")) is False
    assert tool_retry_on(ConnectionError("同上")) is False


class _Request:
    pass


@pytest.mark.asyncio
async def test_normalizer_wraps_unexpected_and_passes_tool_errors():
    normalizer = ToolErrorNormalizerMiddleware()

    async def boom(_request):
        raise KeyError("bug 级意外")

    with pytest.raises(ToolError) as wrapped:
        await normalizer.awrap_tool_call(_Request(), boom)
    assert wrapped.value.retryable is False
    assert "KeyError" in str(wrapped.value)

    async def flaky_timeout(_request):
        raise TimeoutError("source_fetch_timeout")  # direct fetch 的真实形状

    with pytest.raises(ToolError) as wrapped:
        await normalizer.awrap_tool_call(_Request(), flaky_timeout)
    assert wrapped.value.retryable is True  # 传输类瞬时故障仍归可重试

    declared = ToolRequestError("provider 抖动")

    async def declared_failure(_request):
        raise declared

    with pytest.raises(ToolRequestError) as passthrough:
        await normalizer.awrap_tool_call(_Request(), declared_failure)
    assert passthrough.value is declared  # 已声明的可恢复性不被改写


@pytest.mark.asyncio
async def test_normalizer_never_swallows_cancellation():
    normalizer = ToolErrorNormalizerMiddleware()

    async def cancelled(_request):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await normalizer.awrap_tool_call(_Request(), cancelled)


def test_failure_message_carries_the_guard_marker() -> None:
    """ToolLoopGuard 靠 MODEL_FAILURE_MARKER 把模型故障后的引导文本与协议违规区分开;
    _failure_message 一旦绕过该符号拼文案,一次故障会被放大成一整轮新的重试。"""
    from deepresearcher.agents.middleware.retry import MODEL_FAILURE_MARKER

    text = _failure_message("ResearchAgent")(RuntimeError("boom"))
    assert MODEL_FAILURE_MARKER in text
