"""SearchService：与供应商无关的搜索编排。"""

import asyncio
import math
import time
from typing import TYPE_CHECKING, Any

from deepresearcher.config import SearchConfig
from deepresearcher.health import HealthLedger, MemoryHealthLedger
from deepresearcher.tools.errors import (
    ProviderExhaustedError,
    ToolError,
    ToolRequestError,
)
from deepresearcher.tools.transport.http_client import HttpClient
from deepresearcher.tools.web.search.models import SearchResult
from deepresearcher.tools.web.search.providers import build_search_providers

if TYPE_CHECKING:
    from deepresearcher.tools.transport.aliyun import AliyunDtsApi

# 账户级不可用（额度/鉴权）重试同 key 结果不变，冷却时长取长一些。
_ACCOUNT_LEVEL_SECONDS = 900.0

# ProviderExhaustedError 的四个 user_code：恢复前重发结果不变，按账户级不可用上报。
# 限流（rate_limited）只冷却一段固定时长，仍按可重试的请求失败上报。
_ACCOUNT_LEVEL_REASONS = frozenset(
    {"quota_exhausted", "invalid_key", "insufficient_credit", "forbidden"}
)


def _key(name: str) -> str:
    """账本主键带命名空间，与将来 LLM 的 <base_url>:<model> 共用一张表也不撞车。"""
    return f"search:{name}"


class SearchService:
    """按配置顺序尝试候选供应商；确定不可用的那家会被后续请求跳过。"""

    def __init__(
        self,
        config: SearchConfig,
        http_client: HttpClient | None = None,
        *,
        aliyun_client: "AliyunDtsApi | None" = None,
        provider_health: HealthLedger | None = None,
    ):
        self.config = config
        self.http = http_client or HttpClient(config)
        self._health: HealthLedger = provider_health or MemoryHealthLedger()
        # 供应商级并发上限：worker × query 的乘性并发在这里压成对供应商的恒定在飞数。
        self._provider_semaphores: dict[str, asyncio.Semaphore] = {}
        # 构造期即完成初始化：没有可用供应商属于配置问题，必须在这里失败而不是首次检索时。
        self._providers: list[tuple[str, Any]] = build_search_providers(
            config, self.http, aliyun_client
        )

    async def asearch(self, query: str, *, max_results: int | None = None) -> list[SearchResult]:
        limit = max_results or self.config.max_results
        failures: list[str] = []
        errors: list[ToolError] = []
        blocked_reasons: list[str] = []
        for name, provider in self._providers:
            key = _key(name)
            blocked = await self._health.unavailable(key)
            if blocked is not None:
                blocked_reasons.append(blocked.reason)
                failures.append(self._describe_blocked(name, blocked.reason, blocked.retry_after))
                continue
            async with self._semaphore(name):
                # 排队期间可能已被其它并发请求踢出。
                blocked = await self._health.unavailable(key)
                if blocked is not None:
                    blocked_reasons.append(blocked.reason)
                    failures.append(
                        self._describe_blocked(name, blocked.reason, blocked.retry_after)
                    )
                    continue
                try:
                    results = await provider.asearch(query, limit)
                except ProviderExhaustedError as exc:
                    await self._health.trip(key, exc.user_code, _ACCOUNT_LEVEL_SECONDS)
                    errors.append(exc)
                    failures.append(f"{name}：{exc.user_code}")
                    continue
                except ToolError as exc:
                    await self._note_failure(key, exc)
                    errors.append(exc)
                    failures.append(f"{name}：{exc}")
                    continue
                await self._health.record_success(key)
                return results
        message = f"所有搜索供应商都不可用 —— {'；'.join(failures)}"
        # 只有一家时保留原始异常类型与文案，配置单供应商的行为与降级前完全一致。
        if len(errors) == 1:
            raise errors[0]
        exhausted = next((e for e in errors if isinstance(e, ProviderExhaustedError)), None)
        if exhausted is not None:
            raise ProviderExhaustedError(exhausted.user_code, message)
        # 全部被本地冷却短路：手里没有原始异常，按账本记下的原因定性。
        if blocked_reasons and blocked_reasons[0] in _ACCOUNT_LEVEL_REASONS:
            raise ProviderExhaustedError(blocked_reasons[0], message)
        raise ToolRequestError(message, retryable=False)

    async def _note_failure(self, key: str, exc: ToolError) -> None:
        """按失败类型记账：429 走已知恢复点，其余计入连续失败。"""
        reset = getattr(exc, "rate_limit_reset_ts", None)
        if reset:
            await self._health.trip(key, "rate_limited", max(1.0, reset - time.monotonic()))
        else:
            await self._health.record_failure(key, exc.code, self.config.cooldown)

    @staticmethod
    def _describe_blocked(name: str, reason: str, retry_after: float) -> str:
        return f"{name}：{reason}，约 {max(1, math.ceil(retry_after))} 秒后恢复"

    def _semaphore(self, provider: str) -> asyncio.Semaphore:
        semaphore = self._provider_semaphores.get(provider)
        if semaphore is None:
            semaphore = asyncio.Semaphore(self.config.search_max_concurrent_requests)
            self._provider_semaphores[provider] = semaphore
        return semaphore

    @property
    def provider_name(self) -> str:
        """首选供应商；降级时实际响应用哪家看结果里的 content_provider。"""
        return self._providers[0][0]

    @property
    def effective_limit(self) -> int:
        """未显式覆盖时 SearchTool 实际传给供应商的结果上限。"""
        return self.config.max_results
