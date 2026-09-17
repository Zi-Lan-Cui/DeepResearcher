"""统一搜索客户端和供应商无关的候选结果契约。"""

import asyncio
import math
import time
from typing import TYPE_CHECKING

from deepresearcher.config import SearchConfig
from deepresearcher.tools.errors import (
    ProviderExhaustedError,
    ToolConfigurationError,
    ToolRequestError,
)
from deepresearcher.tools.transport.http_client import HttpClient
from deepresearcher.tools.web.search.health import MemoryProviderHealth, ProviderHealth
from deepresearcher.tools.web.search.models import SearchResult

if TYPE_CHECKING:
    from deepresearcher.tools.web.aliyun import AliyunDtsApi

# 账户级不可用（额度/鉴权）的默认熔断窗口：够长以止住 hammer，又能在上游恢复后自愈。
_PROVIDER_HEALTH_SECONDS = 900.0


class SearchClient:
    def __init__(
        self,
        config: SearchConfig,
        http_client: HttpClient | None = None,
        *,
        aliyun_client: "AliyunDtsApi | None" = None,
        provider_health: ProviderHealth | None = None,
    ):
        self.config = config
        self.http = http_client or HttpClient(config)
        # 供应商级断路表：provider -> 单调时钟恢复点。传输层在 429/503 上
        # 记录的恢复时间会写到这里，窗口内的新请求不再出网重复撞墙。
        self._rate_limit_deadlines: dict[str, float] = {}
        # 供应商级并发闸：provider -> Semaphore。worker × query 的乘性并发
        # 在这里收敛为对供应商的恒定在飞请求数。
        self._provider_semaphores: dict[str, asyncio.Semaphore] = {}
        self._aliyun_client = aliyun_client
        # 账户级健康（额度/鉴权）：可跨进程共享；默认进程内存。
        self._health: ProviderHealth = provider_health or MemoryProviderHealth()

    async def asearch(self, query: str, *, max_results: int | None = None) -> list[SearchResult]:
        provider = self.provider_name
        # 账户级熔断优先：同 key 已不可用时连出网都不必。
        open_reason = await self._health.is_open(provider)
        if open_reason:
            raise ProviderExhaustedError(
                open_reason, f"搜索供应商 {provider} 已熔断（{open_reason}），暂停出网。"
            )
        self._raise_if_rate_limited(provider)
        async with self._semaphore(provider):
            # 排队期间断路可能已被别的请求打开；拿到许可后必须复查。
            self._raise_if_rate_limited(provider)
            limit = max_results or self.config.max_results
            try:
                return await self._provider().asearch(query, limit)
            except ProviderExhaustedError as exc:
                # 鉴权/额度型：打开健康位，让本 worker 后续（乃至跨 worker）快速失败。
                await self._health.trip(provider, exc.user_code, _PROVIDER_HEALTH_SECONDS)
                raise
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
        from deepresearcher.tools.web.search.providers import (
            AliyunSearchProvider,
            BaiduSearchProvider,
            SerpApiSearchProvider,
            TavilySearchProvider,
        )

        providers = {
            "baidu": (self.config.baidu_api_key, BaiduSearchProvider),
            "tavily": (self.config.tavily_api_key, TavilySearchProvider),
            "serpapi": (self.config.serpapi_api_key, SerpApiSearchProvider),
        }
        if self.config.provider == "aliyun":
            if self._aliyun_client is None:
                raise ToolConfigurationError("已选择 aliyun，但未初始化阿里云 DTS AI 客户端")
            return AliyunSearchProvider(self._aliyun_client)
        if self.config.provider != "auto":
            key, provider_type = providers[self.config.provider]
            if not key:
                raise ToolConfigurationError(f"已选择 {self.config.provider}，但未配置对应 API Key")
            return provider_type(self.config, self.http)
        for key, provider_type in providers.values():
            if key:
                return provider_type(self.config, self.http)
        raise ToolConfigurationError("未配置 BAIDU_API_KEY、TAVILY_API_KEY 或 SERPAPI_API_KEY")
