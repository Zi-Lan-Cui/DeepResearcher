"""消息 token 预算的轻量估算。"""

from collections.abc import Sequence

from langchain_core.messages import BaseMessage

from deepsearch_agent.evidence.tokens import TokenEstimator, get_token_estimator


def message_text(message: BaseMessage) -> str:
    """把消息中用于计数的内容规范化为字符串。"""
    content = message.content
    return content if isinstance(content, str) else str(content)


class MessageBudget:
    """使用缓存的 tokenizer 估算消息列表大小。"""

    def __init__(self, estimator: TokenEstimator | None = None) -> None:
        self.estimator = estimator or get_token_estimator()

    def count(self, messages: Sequence[BaseMessage]) -> int:
        return sum(self.estimator.count(message_text(message)) for message in messages)
