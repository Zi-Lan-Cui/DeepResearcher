"""跨层共享的应用错误契约与错误文本钳位。"""

# 错误文本进事件 payload、回执与日志前的统一上限。
ERROR_CLIP_CHARS = 500


def clip_text(text: str | None, limit: int = ERROR_CLIP_CHARS) -> str:
    """定长截断并留痕:超长时尾部以标记明示,避免读者把片段当全文。"""
    if not text:
        return ""
    return text if len(text) <= limit else text[: limit - 8] + "…[截断]"


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
