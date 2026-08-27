"""Evidence 文本的 token 估算器。

Tokenizer 是进程级只读资源，使用工厂缓存，避免每个来源重复初始化。
估算器只负责计数；正文仍使用原始字符串传递，保证 quote 可以逐字校验。
"""

from functools import lru_cache
from typing import Protocol

import tiktoken


class TokenEstimator(Protocol):
    def count(self, text: str) -> int: ...


class TiktokenEstimator:
    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self._encoding = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        return len(self._encoding.encode(text, disallowed_special=()))


@lru_cache(maxsize=8)
def get_token_estimator(encoding_name: str = "cl100k_base") -> TokenEstimator:
    """返回缓存的 tokenizer estimator；同一编码只初始化一次。"""
    return TiktokenEstimator(encoding_name)
