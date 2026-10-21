"""文本 token 估算：中间件压缩判断与来源读取窗口共用的本地计数。

与 execution.usage 的提供商真实计费是两套标准：本模块估算决定何时压缩与截窗，
后者实收决定预算熔断。估算器只计数，正文原样传递，保证 quote 可逐字校验。
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
    """返回缓存的估算器；同一编码只初始化一次。"""
    return TiktokenEstimator(encoding_name)
