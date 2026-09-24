"""文本 token 估算：中间件压缩判断与来源读取窗口共用的本地计数。

与 service.usage 的提供商真实计费是两套口径:本模块做估算,决定何时
压缩与截窗;service.usage 做实收,决定预算熔断。两者各自独立。Tokenizer 是进程级只读资源,
工厂缓存避免每个调用方重复初始化;估算器只负责计数,正文仍原样传递,
保证 quote 可以逐字校验。"""

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
