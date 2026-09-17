"""LLM 装配与调用层错误。"""

from deepresearcher.errors import AgentError


class LLMError(AgentError):
    code = "llm_error"


class LLMConfigurationError(LLMError):
    """研究链所需的模型能力未在应用装配期提供。"""

    code = "llm_configuration"


class LLMTimeoutError(LLMError):
    """LLM 请求超时，可按统一策略重试。"""

    code = "llm_timeout"
    retryable = True


class LLMUnavailableError(LLMError):
    """LLM 网关**账户级不可用**：key 无效 / 无余额 / 硬限流。

    连模型都调不动 → 无法"提醒 agent 收尾"，必须快速失败并把 user_code 透到前端。
    """

    code = "llm_unavailable"
    retryable = False

    def __init__(self, user_code: str, message: str):
        super().__init__(message)
        self.user_code = user_code  # invalid_key / forbidden / insufficient_credit / rate_limited


# openai 等 SDK 的 APIStatusError 带 .status_code；映射为稳定 user_code（不可自愈的那类）。
_LLM_FATAL_STATUS = {401: "invalid_key", 402: "insufficient_credit", 403: "forbidden"}


def classify_llm_error(exc: BaseException) -> str | None:
    """把传输层抛来的原始异常归类为 llm 不可用 user_code；非致命返回 None。"""
    status = getattr(exc, "status_code", None)
    if status in _LLM_FATAL_STATUS:
        return _LLM_FATAL_STATUS[status]
    if status == 429:
        # RateLimit 通常已被 transport retry 消化；冒泡到这里=重试后仍不可用。
        return "rate_limited"
    return None
