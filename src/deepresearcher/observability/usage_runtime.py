"""服务无关的 run 用量上下文:agent 与工具经 ContextVar 读取记账与预算。"""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Protocol

from deepresearcher.config import LLMConfig
from deepresearcher.observability.logging_config import get_logger

logger = get_logger("deepresearcher.observability.usage_runtime")


class UsageRecorder(Protocol):
    """Minimal execution-owned capability exposed to engine code."""

    async def enforce_budget(self, run_id: str, config: LLMConfig) -> None: ...

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
    ) -> None: ...


@dataclass(frozen=True)
class UsageRuntime:
    run_id: str
    store: UsageRecorder
    config: LLMConfig


class UsageBudgetExceeded(RuntimeError):
    """The current run reached one of its configured LLM usage limits."""


_runtime: ContextVar[UsageRuntime | None] = ContextVar("deepresearcher_usage_runtime", default=None)


def bind_usage_runtime(runtime: UsageRuntime) -> Token:
    return _runtime.set(runtime)


def reset_usage_runtime(token: Token) -> None:
    _runtime.reset(token)


def current_usage_runtime() -> UsageRuntime | None:
    return _runtime.get()


async def enforce_usage_budget() -> None:
    """Check the active run budget; no-op outside a service execution context."""

    runtime = current_usage_runtime()
    if runtime is not None:
        await runtime.store.enforce_budget(runtime.run_id, runtime.config)


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
    except Exception:  # noqa: BLE001 - accounting must not break tool delivery
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
    except Exception:  # noqa: BLE001 - cache accounting does not alter cache semantics
        logger.warning("cache_usage_record_failed run_id=%s", runtime.run_id, exc_info=True)
