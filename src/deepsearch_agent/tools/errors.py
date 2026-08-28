from deepsearch_agent.errors import AgentError


class ToolError(AgentError):
    """所有外部工具错误的基类。"""


class ToolConfigurationError(ToolError):
    """工具缺少必要配置。"""

    code = "tool_configuration"


class ToolRequestError(ToolError):
    """请求失败，包含最终可诊断原因。"""

    code = "tool_request"
    retryable = True
    # 传输层在 429/503 时写入的本地单调时钟恢复点；None 表示不是配额型失败。
    rate_limit_reset_ts: float | None = None


class ToolParseError(ToolError):
    """响应或文档解析失败。"""

    code = "tool_parse"


class SourceUnavailableError(ToolError):
    """来源可访问但无法取得可验证正文，例如验证码页、登录墙或动态空壳。"""

    code = "source_unavailable"

    def __init__(self, reason_code: str, message: str):
        super().__init__(message)
        self.reason_code = reason_code
