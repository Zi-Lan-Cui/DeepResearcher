"""网页抓取编排：Provider 顺序尝试与可恢复降级。

正文级缓存不在本层——由 SourceReaderTool 经 ResearchMaterialStore.resolve_fetch
命中/回写；material_fetch_key 只是把"可跨 Run 复用"的身份算给 reader 用。
"""

import asyncio

from deepresearcher.tools.cache_keys import canonical_url, semantic_cache_key
from deepresearcher.tools.web.fetch.models import SourceDocument
from deepresearcher.tools.web.fetch.protocol import FetchProvider


class FetchService:
    """按配置顺序尝试 FetchProvider，并向上游提供单一抓取接口。"""

    name = "fetch_service"

    def __init__(
        self,
        providers: list[FetchProvider],
        *,
        fetch_policy_version: str = "public-fetch-v1",
        parser_version: str = "parser-v1",
    ):
        if not providers:
            raise ValueError("FetchService 至少需要一个 FetchProvider。")
        self.providers = list(providers)
        self.fetch_policy_version = fetch_policy_version
        self.parser_version = parser_version

    def material_fetch_key(self, url: str) -> str:
        """返回可跨 Run 复用的正文身份：URL 规范串 + provider 序列 + 策略/解析版本；任一版本变化即自然失效。"""

        normalized_url = canonical_url(url)
        if not normalized_url:
            return ""
        return semantic_cache_key(
            normalized_url,
            tuple(provider.name for provider in self.providers),
            self.fetch_policy_version,
            self.parser_version,
        )

    async def afetch(
        self,
        url: str,
        *,
        fetch_timeout: float | None = None,
        parse_timeout: float | None = None,
    ) -> SourceDocument:
        """按配置顺序尝试各 Provider：成功即归因 retrieval_method，全败保留原异常语义。"""
        last_error: Exception | None = None
        last_failure: SourceDocument | None = None
        for provider in self.providers:
            try:
                document = await provider.afetch(
                    url,
                    fetch_timeout=fetch_timeout,
                    parse_timeout=parse_timeout,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # Provider 失败由下一项接管；最终仍保留原异常语义。
                last_error = exc
                last_failure = None
                continue
            if document.get("status") == "completed" and not document.get("error"):
                document.setdefault("retrieval_method", f"{provider.name}_fetch")
                return document
            last_failure = document
            last_error = None
        if last_error is not None:
            raise last_error
        if last_failure is not None:
            return last_failure
        raise RuntimeError("所有 FetchProvider 均未返回结果。")
