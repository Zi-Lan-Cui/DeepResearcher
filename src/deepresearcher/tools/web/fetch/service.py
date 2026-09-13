"""网页抓取编排：缓存、Provider 顺序和可恢复降级。"""

import asyncio
from typing import cast

from deepresearcher.tools.cache import CacheValue, NoOpToolCache, ToolCache
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
        tool_cache: ToolCache | None = None,
        cache_ttl_seconds: int = 0,
        fetch_policy_version: str = "public-fetch-v1",
        parser_version: str = "parser-v1",
    ):
        if not providers:
            raise ValueError("FetchService 至少需要一个 FetchProvider。")
        self.providers = list(providers)
        self.tool_cache = tool_cache or NoOpToolCache()
        self.cache_ttl_seconds = cache_ttl_seconds
        self.fetch_policy_version = fetch_policy_version
        self.parser_version = parser_version

    async def afetch(
        self,
        url: str,
        *,
        fetch_timeout: float | None = None,
        parse_timeout: float | None = None,
    ) -> SourceDocument:
        normalized_url = canonical_url(url)
        if not normalized_url or not normalized_url.startswith(("http://", "https://")):
            return await self._fetch_uncached(
                url, fetch_timeout=fetch_timeout, parse_timeout=parse_timeout
            )

        async def compute() -> CacheValue:
            document = await self._fetch_uncached(
                url, fetch_timeout=fetch_timeout, parse_timeout=parse_timeout
            )
            return CacheValue(
                value=dict(document),
                content_hash=document.get("content_hash"),
                metrics={"saved_external_requests": 1},
                cacheable=document.get("status") == "completed" and not document.get("error"),
            )

        provider_chain = tuple(provider.name for provider in self.providers)
        cached = await self.tool_cache.get_or_compute(
            "fetch",
            semantic_cache_key(
                normalized_url,
                provider_chain,
                self.fetch_policy_version,
                self.parser_version,
            ),
            ttl_seconds=self.cache_ttl_seconds,
            schema_version=self.parser_version,
            compute=compute,
        )
        document = cast(SourceDocument, dict(cached.value))
        document["source_url"] = url
        document["cache_hit"] = cached.hit
        if cached.hit:
            document["fetch_duration_ms"] = 0
            document["parse_duration_ms"] = 0
        return document

    async def _fetch_uncached(
        self,
        url: str,
        *,
        fetch_timeout: float | None,
        parse_timeout: float | None,
    ) -> SourceDocument:
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
