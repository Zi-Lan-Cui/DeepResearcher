"""Run-scoped capacity control and usage/cost accounting."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from contextvars import ContextVar, Token
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from time import monotonic
from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from sqlalchemy import case, func, select, update

from deepsearch_agent.config import LLMConfig
from deepsearch_agent.observability.tracing.context import new_id
from deepsearch_agent.service.persistence.models import Run, RunUsage

logger = logging.getLogger("deepsearch_agent.service.usage")


class CapacityGate:
    def __init__(self, limit: int) -> None:
        self._semaphore = asyncio.Semaphore(max(1, limit))
        self._active = 0
        self._lock = asyncio.Lock()

    async def acquire(self) -> int:
        await self._semaphore.acquire()
        async with self._lock:
            self._active += 1
            return self._active

    async def release(self) -> None:
        async with self._lock:
            self._active = max(0, self._active - 1)
        self._semaphore.release()


class ProviderRateLimiter:
    """每 Worker 共享的 60 秒滑动窗口 RPM/TPM 限制器。"""

    def __init__(
        self,
        *,
        requests_per_minute: int,
        tokens_per_minute: int,
        window_seconds: float = 60.0,
    ) -> None:
        self._rpm = max(0, requests_per_minute)
        self._tpm = max(0, tokens_per_minute)
        self._window_seconds = max(0.01, window_seconds)
        self._requests: deque[float] = deque()
        self._tokens: deque[tuple[float, int]] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self, estimated_tokens: int) -> None:
        if not self._rpm and not self._tpm:
            return
        reservation = min(max(1, estimated_tokens), self._tpm) if self._tpm else 0
        while True:
            async with self._lock:
                now = monotonic()
                cutoff = now - self._window_seconds
                while self._requests and self._requests[0] <= cutoff:
                    self._requests.popleft()
                while self._tokens and self._tokens[0][0] <= cutoff:
                    self._tokens.popleft()
                request_ok = not self._rpm or len(self._requests) < self._rpm
                token_ok = (
                    not self._tpm
                    or sum(value for _, value in self._tokens) + reservation <= self._tpm
                )
                if request_ok and token_ok:
                    self._requests.append(now)
                    if self._tpm:
                        self._tokens.append((now, reservation))
                    return
                deadlines: list[float] = []
                if not request_ok and self._requests:
                    deadlines.append(self._requests[0] + self._window_seconds)
                if not token_ok and self._tokens:
                    deadlines.append(self._tokens[0][0] + self._window_seconds)
                delay = max(0.01, min(deadlines) - now) if deadlines else 0.01
            await asyncio.sleep(delay)


@dataclass(frozen=True)
class UsageRuntime:
    run_id: str
    store: "UsageStore"
    config: LLMConfig


class UsageBudgetExceeded(RuntimeError):
    """本次运行已达配置的 LLM 用量上限。"""


_runtime: ContextVar[UsageRuntime | None] = ContextVar("deepsearch_usage_runtime", default=None)


def bind_usage_runtime(runtime: UsageRuntime) -> Token:
    return _runtime.set(runtime)


def reset_usage_runtime(token: Token) -> None:
    _runtime.reset(token)


def current_usage_runtime() -> UsageRuntime | None:
    return _runtime.get()


async def enforce_usage_budget() -> None:
    """检查当前 run 的用量预算；非服务执行上下文中自动跳过。"""
    runtime = current_usage_runtime()
    if runtime is not None:
        await runtime.store.enforce_budget(runtime.run_id, runtime.config)


class UsageStore:
    def __init__(self, session_factory: Any) -> None:
        self._session_factory = session_factory

    async def record(
        self,
        *,
        run_id: str,
        category: str,
        component: str,
        status: str,
        provider: str | None = None,
        model: str | None = None,
        estimated: bool = False,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_input_tokens: int = 0,
        cost_usd: Decimal = Decimal("0"),
        duration_ms: int = 0,
        active: int = 0,
        detail: dict[str, Any] | None = None,
    ) -> None:
        async with self._session_factory() as session:
            session.add(
                RunUsage(
                    id=new_id("usage"),
                    run_id=run_id,
                    category=category,
                    component=component,
                    provider=provider,
                    model=model,
                    status=status,
                    usage_estimated=estimated,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cached_input_tokens=cached_input_tokens,
                    cost_usd=cost_usd,
                    duration_ms=duration_ms,
                    detail_json=detail,
                )
            )
            values: dict[str, Any] = {}
            if category == "llm":
                values = {
                    "llm_call_count": Run.llm_call_count + 1,
                    "input_tokens": Run.input_tokens + input_tokens,
                    "output_tokens": Run.output_tokens + output_tokens,
                    "cached_input_tokens": Run.cached_input_tokens + cached_input_tokens,
                    "peak_llm_concurrency": case(
                        (Run.peak_llm_concurrency < active, active),
                        else_=Run.peak_llm_concurrency,
                    ),
                    "estimated_cost_usd": Run.estimated_cost_usd + cost_usd,
                }
            elif category in {"search", "fetch"}:
                values = {"external_request_count": Run.external_request_count + 1}
            elif category == "cache" and status == "hit":
                metrics = detail or {}
                values = {
                    "cache_hit_count": Run.cache_hit_count + 1,
                    "saved_external_request_count": (
                        Run.saved_external_request_count
                        + int(metrics.get("saved_external_requests", 0) or 0)
                    ),
                    "saved_llm_call_count": (
                        Run.saved_llm_call_count + int(metrics.get("saved_llm_calls", 0) or 0)
                    ),
                    "saved_tokens": (Run.saved_tokens + int(metrics.get("saved_tokens", 0) or 0)),
                    "saved_cost_usd": (
                        Run.saved_cost_usd + Decimal(str(metrics.get("saved_cost_usd", 0) or 0))
                    ),
                }
            if values:
                await session.execute(update(Run).where(Run.id == run_id).values(**values))
            await session.commit()

    async def enforce_budget(self, run_id: str, config: LLMConfig) -> None:
        """在新的 LLM 请求前做软限额检查。

        单次请求的真实 token 只能在返回后获得，因此最多会超出一次
        已在途请求的用量。并发下的精硬扣减留给后续配额服务。
        """
        if not any(
            (
                config.max_tokens_per_run,
                config.max_cost_usd_per_run,
                config.max_cost_usd_per_user_daily,
                config.max_cost_usd_platform_hourly,
                config.max_cost_usd_platform_daily,
            )
        ):
            return
        async with self._session_factory() as session:
            run = await session.get(Run, run_id)
            if run is None:
                return
            total_tokens = run.input_tokens + run.output_tokens
            if config.max_tokens_per_run and total_tokens >= config.max_tokens_per_run:
                raise UsageBudgetExceeded("run_token_budget_exhausted")
            if config.max_cost_usd_per_run and run.estimated_cost_usd >= Decimal(
                str(config.max_cost_usd_per_run)
            ):
                raise UsageBudgetExceeded("run_cost_budget_exhausted")
            now = datetime.now(timezone.utc)
            day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            hour_start = now - timedelta(hours=1)
            if config.max_cost_usd_per_user_daily:
                user_cost = await session.scalar(
                    select(func.coalesce(func.sum(RunUsage.cost_usd), 0))
                    .join(Run, Run.id == RunUsage.run_id)
                    .where(
                        Run.user_id == run.user_id,
                        RunUsage.category == "llm",
                        RunUsage.created_at >= day_start,
                    )
                )
                if Decimal(user_cost or 0) >= Decimal(str(config.max_cost_usd_per_user_daily)):
                    raise UsageBudgetExceeded("user_daily_cost_budget_exhausted")
            if config.max_cost_usd_platform_hourly:
                hourly_cost = await session.scalar(
                    select(func.coalesce(func.sum(RunUsage.cost_usd), 0)).where(
                        RunUsage.category == "llm", RunUsage.created_at >= hour_start
                    )
                )
                if Decimal(hourly_cost or 0) >= Decimal(str(config.max_cost_usd_platform_hourly)):
                    raise UsageBudgetExceeded("platform_hourly_cost_budget_exhausted")
            if config.max_cost_usd_platform_daily:
                daily_cost = await session.scalar(
                    select(func.coalesce(func.sum(RunUsage.cost_usd), 0)).where(
                        RunUsage.category == "llm", RunUsage.created_at >= day_start
                    )
                )
                if Decimal(daily_cost or 0) >= Decimal(str(config.max_cost_usd_platform_daily)):
                    raise UsageBudgetExceeded("platform_daily_cost_budget_exhausted")


class RunUsageCallback(AsyncCallbackHandler):
    """Capture every provider attempt, including Agent and structured calls."""

    def __init__(
        self,
        *,
        run_id: str,
        store: UsageStore,
        gate: CapacityGate,
        rate_limiter: ProviderRateLimiter,
        config: LLMConfig,
    ) -> None:
        self.raise_error = True
        self._run_id = run_id
        self._store = store
        self._gate = gate
        self._rate_limiter = rate_limiter
        self._config = config
        self._starts: dict[UUID, tuple[float, int, int, dict[str, Any]]] = {}
        self._active = 0
        self._active_lock = asyncio.Lock()

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        metadata: dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> None:
        try:
            await self._store.enforce_budget(self._run_id, self._config)
        except UsageBudgetExceeded:
            raise
        except Exception:  # noqa: BLE001 - 计量存储短暂故障不阻断 provider
            logger.warning("usage_budget_check_failed run_id=%s", self._run_id, exc_info=True)
        prompt_chars = sum(len(str(message.content)) for batch in messages for message in batch)
        estimated_tokens = max(1, prompt_chars // 4) + self._config.provider_output_token_reserve
        await self._rate_limiter.acquire(estimated_tokens)
        await self._gate.acquire()
        async with self._active_lock:
            self._active += 1
            active = self._active
        self._starts[run_id] = (monotonic(), active, prompt_chars, metadata or {})

    async def on_llm_end(self, response: Any, *, run_id: UUID, **_kwargs: Any) -> None:
        started, active, prompt_chars, metadata = self._starts.pop(run_id, (monotonic(), 1, 0, {}))
        try:
            input_tokens, output_tokens, cached_tokens, actual = _response_usage(response)
            if not actual:
                input_tokens = max(1, prompt_chars // 4)
                output_tokens = max(0, _response_chars(response) // 4)
            await self._record(
                status="completed",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_tokens=cached_tokens,
                estimated=not actual,
                started=started,
                active=active,
                metadata=metadata,
            )
        finally:
            await self._release()

    async def on_llm_error(self, error: BaseException, *, run_id: UUID, **_kwargs: Any) -> None:
        started, active, prompt_chars, metadata = self._starts.pop(run_id, (monotonic(), 1, 0, {}))
        try:
            await self._record(
                status="failed",
                input_tokens=max(1, prompt_chars // 4),
                output_tokens=0,
                cached_tokens=0,
                estimated=True,
                started=started,
                active=active,
                metadata=metadata,
                detail={"error_type": type(error).__name__},
            )
        finally:
            await self._release()

    async def _release(self) -> None:
        async with self._active_lock:
            self._active = max(0, self._active - 1)
        await self._gate.release()

    async def _record(
        self,
        *,
        status: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int,
        estimated: bool,
        started: float,
        active: int,
        metadata: dict[str, Any],
        detail: dict[str, Any] | None = None,
    ) -> None:
        try:
            await self._store.record(
                run_id=self._run_id,
                category="llm",
                component=str(metadata.get("langgraph_node") or "model"),
                provider=str(metadata.get("ls_provider") or "") or None,
                model=str(metadata.get("ls_model_name") or self._config.model) or None,
                status=status,
                estimated=estimated,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=cached_tokens,
                cost_usd=_cost(self._config, input_tokens, output_tokens, cached_tokens),
                duration_ms=round((monotonic() - started) * 1000),
                active=active,
                detail={"price_version": self._config.price_version, **(detail or {})},
            )
        except Exception:  # noqa: BLE001 - 计量故障不能破坏研究交付
            logger.warning("usage_record_failed run_id=%s", self._run_id, exc_info=True)


async def record_external_request(
    *, category: str, status: str, duration_ms: int, detail: dict[str, Any] | None = None
) -> None:
    runtime = current_usage_runtime()
    if runtime is None:
        return
    try:
        await runtime.store.record(
            run_id=runtime.run_id,
            category=category,
            component=category,
            status=status,
            duration_ms=duration_ms,
            detail=detail,
        )
    except Exception:  # noqa: BLE001
        logger.warning("external_usage_record_failed run_id=%s", runtime.run_id, exc_info=True)


async def record_cache_event(
    *, namespace: str, status: str, detail: dict[str, Any] | None = None
) -> None:
    runtime = current_usage_runtime()
    if runtime is None:
        return
    try:
        await runtime.store.record(
            run_id=runtime.run_id,
            category="cache",
            component=namespace,
            status=status,
            detail=detail,
        )
    except Exception:  # noqa: BLE001 - 缓存计量不改变工具语义
        logger.warning("cache_usage_record_failed run_id=%s", runtime.run_id, exc_info=True)


def _response_usage(response: Any) -> tuple[int, int, int, bool]:
    input_tokens = output_tokens = cached_tokens = 0
    actual = False
    for batch in getattr(response, "generations", ()) or ():
        for generation in batch:
            message = getattr(generation, "message", None)
            usage = getattr(message, "usage_metadata", None) or {}
            if usage:
                message_input = int(usage.get("input_tokens") or 0)
                message_output = int(usage.get("output_tokens") or 0)
                actual = actual or message_input > 0 or message_output > 0
                input_tokens += message_input
                output_tokens += message_output
                details = usage.get("input_token_details") or {}
                cached_tokens += int(details.get("cache_read") or details.get("cached_tokens") or 0)
    token_usage = (getattr(response, "llm_output", None) or {}).get("token_usage") or {}
    if token_usage and not actual:
        input_tokens = int(token_usage.get("prompt_tokens") or 0)
        output_tokens = int(token_usage.get("completion_tokens") or 0)
        cached_tokens = int(
            (token_usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        )
        actual = input_tokens > 0 or output_tokens > 0
    return input_tokens, output_tokens, cached_tokens, actual


def _response_chars(response: Any) -> int:
    return sum(
        len(str(getattr(generation, "text", "") or ""))
        for batch in getattr(response, "generations", ()) or ()
        for generation in batch
    )


def _cost(config: LLMConfig, input_tokens: int, output_tokens: int, cached: int) -> Decimal:
    uncached = max(0, input_tokens - cached)
    total = (
        Decimal(uncached) * Decimal(str(config.input_usd_per_million))
        + Decimal(cached) * Decimal(str(config.cached_input_usd_per_million))
        + Decimal(output_tokens) * Decimal(str(config.output_usd_per_million))
    ) / Decimal(1_000_000)
    return total.quantize(Decimal("0.00000001"))
