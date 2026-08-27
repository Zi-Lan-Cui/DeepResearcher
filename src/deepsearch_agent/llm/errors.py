"""LLM 装配与调用层错误。"""

from deepsearch_agent.errors import AgentError


class LLMError(AgentError):
    code = "llm_error"


class LLMConfigurationError(LLMError):
    """研究链所需的模型能力未在应用装配期提供。"""

    code = "llm_configuration"


class LLMTimeoutError(LLMError):
    """LLM 请求超时，可按统一策略重试。"""

    code = "llm_timeout"
    retryable = True
