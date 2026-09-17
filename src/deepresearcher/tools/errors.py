from deepresearcher.errors import AgentError


class ToolError(AgentError):
    """所有外部工具错误的基类。"""


class ProviderExhaustedError(ToolError):
    """数据提供方**账户级**不可用：鉴权失败 / 额度耗尽 / key 无效。

    重试单条无意义（同 key 同结果），应触发熔断并让上层快速收尾。与
    "某个来源 401/403（付费墙/登录墙）" 区分开——后者是 per-source 的
    SourceUnavailable/ToolRequestError，不代表 provider 整体挂掉。
    """

    code = "provider_exhausted"
    retryable = False

    def __init__(self, user_code: str, message: str):
        super().__init__(message)
        self.user_code = user_code  # invalid_key / forbidden / insufficient_credit / quota_exhausted


class ToolConfigurationError(ToolError):
    """工具缺少必要配置。"""

    code = "tool_configuration"


class ToolRequestError(ToolError):
    """请求失败，包含最终可诊断原因。"""

    code = "tool_request"
    retryable = True
    # 传输层在 429/503 时写入的本地单调时钟恢复点；None 表示不是配额型失败。
    rate_limit_reset_ts: float | None = None


class UnsafeUrlError(ToolRequestError):
    """A source URL violates the public-network fetching policy."""

    code = "unsafe_url"
    retryable = False


class ToolParseError(ToolError):
    """响应或文档解析失败。"""

    code = "tool_parse"


class SourceUnavailableError(ToolError):
    """来源可访问但无法取得可验证正文，例如验证码页、登录墙或动态空壳。"""

    code = "source_unavailable"

    def __init__(self, reason_code: str, message: str):
        super().__init__(message)
        self.reason_code = reason_code
