"""统一搜索客户端和供应商无关的候选结果契约。"""

from typing import NotRequired, TypedDict

from deepsearch_agent.config import SearchConfig
from deepsearch_agent.tools.errors import ToolConfigurationError
from deepsearch_agent.tools.http_client import HttpClient


class SearchResult(TypedDict, total=False):
    title: str
    url: str
    snippet: NotRequired[str]
    raw_content: NotRequired[str]
    content_provider: NotRequired[str]
    score: NotRequired[float]
    published_at: NotRequired[str]


class SearchClient:
    def __init__(self, config: SearchConfig, http_client: HttpClient | None = None):
        self.config = config
        self.http = http_client or HttpClient(config)

    async def asearch(self, query: str, *, max_results: int | None = None) -> list[SearchResult]:
        limit = max_results or self.config.max_results
        return await self._provider().asearch(query, limit)

    @property
    def provider_name(self) -> str:
        """返回本次配置实际选择的搜索供应商，便于诊断有效配置。"""
        if self.config.provider != "auto":
            return self.config.provider
        if self.config.baidu_api_key:
            return "baidu"
        if self.config.tavily_api_key:
            return "tavily"
        if self.config.serpapi_api_key:
            return "serpapi"
        return "unconfigured"

    @property
    def effective_limit(self) -> int:
        """返回未显式覆盖时 SearchTool 实际传给供应商的结果上限。"""
        return self.config.max_results

    def _provider(self):
        from deepsearch_agent.tools.search_providers import (
            BaiduSearchProvider,
            SerpApiSearchProvider,
            TavilySearchProvider,
        )
        providers = {
            "baidu": (self.config.baidu_api_key, BaiduSearchProvider),
            "tavily": (self.config.tavily_api_key, TavilySearchProvider),
            "serpapi": (self.config.serpapi_api_key, SerpApiSearchProvider),
        }
        if self.config.provider != "auto":
            key, provider_type = providers[self.config.provider]
            if not key:
                raise ToolConfigurationError(f"已选择 {self.config.provider}，但未配置对应 API Key")
            return provider_type(self.config, self.http)
        for key, provider_type in providers.values():
            if key:
                return provider_type(self.config, self.http)
        raise ToolConfigurationError("未配置 BAIDU_API_KEY、TAVILY_API_KEY 或 SERPAPI_API_KEY")
