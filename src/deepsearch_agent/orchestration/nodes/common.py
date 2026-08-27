"""流程节点共享的确定性判断。"""


def content_text(response: object) -> str:
    """将 ChatModel 响应内容稳定转换为文本。"""
    content = getattr(response, "content", response)
    return content if isinstance(content, str) else str(content)
