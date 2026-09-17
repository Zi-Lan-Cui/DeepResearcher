"""网页抓取编排：缓存、Provider 顺序和可恢复降级。"""

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
        """返回可跨 Run 复用的正文身份；与现有 ToolCache 版本语义一致。"""

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
        normalized_url = canonical_url(url)
        if not normalized_url or not normalized_url.startswith(("http://", "https://")):
            return await self._fetch_uncached(
                url, fetch_timeout=fetch_timeout, parse_timeout=parse_timeout
            )
        return await self._fetch_uncached(
            url, fetch_timeout=fetch_timeout, parse_timeout=parse_timeout
        )

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
