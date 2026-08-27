"""跨层共享的应用错误契约。"""


class AgentError(RuntimeError):
    """可被运行边界识别的应用错误。"""

    code = "agent_error"
    retryable = False

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        retryable: bool | None = None,
        detail: str = "",
    ):
        super().__init__(message)
        self.code = code or type(self).code
        self.retryable = type(self).retryable if retryable is None else retryable
        self.detail = detail
