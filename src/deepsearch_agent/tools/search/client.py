"""统一搜索客户端和供应商无关的候选结果契约。"""

import asyncio
import math
import time

from deepsearch_agent.config import SearchConfig
from deepsearch_agent.tools.errors import ToolConfigurationError, ToolRequestError
from deepsearch_agent.tools.search.models import SearchResult
from deepsearch_agent.tools.transport.http_client import HttpClient


class SearchClient:
    def __init__(self, config: SearchConfig, http_client: HttpClient | None = None):
        self.config = config
        self.http = http_client or HttpClient(config)
        # 供应商级断路表：provider -> 单调时钟恢复点。传输层在 429/503 上
        # 记录的恢复时间会写到这里，窗口内的新请求不再出网重复撞墙。
        self._rate_limit_deadlines: dict[str, float] = {}
        # 供应商级并发闸：provider -> Semaphore。worker × query 的乘性并发
        # 在这里收敛为对供应商的恒定在飞请求数。
        self._provider_semaphores: dict[str, asyncio.Semaphore] = {}

    async def asearch(self, query: str, *, max_results: int | None = None) -> list[SearchResult]:
        provider = self.provider_name
        self._raise_if_rate_limited(provider)
        async with self._semaphore(provider):
            # 排队期间断路可能已被别的请求打开；拿到许可后必须复查。
            self._raise_if_rate_limited(provider)
            limit = max_results or self.config.max_results
            try:
                return await self._provider().asearch(query, limit)
            except ToolRequestError as exc:
                reset = getattr(exc, "rate_limit_reset_ts", None)
                if reset:
                    self._rate_limit_deadlines[provider] = max(
                        self._rate_limit_deadlines.get(provider, 0.0), reset
                    )
                raise

    def _raise_if_rate_limited(self, provider: str) -> None:
        deadline = self._rate_limit_deadlines.get(provider, 0.0)
        now = time.monotonic()
        if now < deadline:
            # retryable=False：本窗口内立即重试没有意义，把决策让给模型换策略。
            raise ToolRequestError(
                f"搜索供应商 {provider} 已被限流，跳过远程重试；"
                f"约 {max(1, math.ceil(deadline - now))} 秒后恢复。"
                "请基于已读取来源完成当前方向，或如实上报证据不足。",
                retryable=False,
            )

    def _semaphore(self, provider: str) -> asyncio.Semaphore:
        semaphore = self._provider_semaphores.get(provider)
        if semaphore is None:
            semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
            self._provider_semaphores[provider] = semaphore
        return semaphore

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
        from deepsearch_agent.tools.search.providers import (
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
