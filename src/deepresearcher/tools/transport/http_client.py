"""异步 HTTP 客户端：统一处理浏览器 TLS 指纹、超时和重试。"""

import asyncio
import email.utils
import random
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, cast
from urllib.parse import urljoin

from curl_cffi import CurlOpt
from curl_cffi.requests import AsyncSession, Response
from curl_cffi.requests.exceptions import RequestException, Timeout
from curl_cffi.requests.impersonate import DEFAULT_CHROME, DEFAULT_FIREFOX, DEFAULT_SAFARI

from deepresearcher.config import SearchConfig
from deepresearcher.observability.usage_runtime import record_external_request
from deepresearcher.tools.errors import ProviderExhaustedError, ToolRequestError
from deepresearcher.tools.transport.url_guard import PublicUrlGuard

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
# 429/503 是“配额/限流”型失败：除了退避重试，还要给调用方留下跨请求的恢复点。
_RATE_LIMITED_STATUS = {429, 503}
# 搜索提供方（同一 key）账户级不可自愈状态：重试单条无意义 → 抛 ProviderExhausted 触熔断。
# 只对 request_kind=="search" 生效；fetch 的 401/403 是单来源（付费墙/登录墙），非 provider 挂。
_SEARCH_PROVIDER_FATAL = {401: "invalid_key", 402: "insufficient_credit", 403: "forbidden", 432: "quota_exhausted"}
_DEFAULT_RATE_LIMIT_WINDOW_SECONDS = 30.0
_IMPERSONATE_TARGETS = (DEFAULT_CHROME, DEFAULT_FIREFOX, DEFAULT_SAFARI)
_JITTER_RATIO = 0.25
_REDIRECT_STATUS = {301, 302, 303, 307, 308}
_MAX_PUBLIC_REDIRECTS = 5


def parse_retry_after(value: str | None) -> float | None:
    """解析 Retry-After 响应头：delta-seconds 或 HTTP-date 两种格式。"""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return float(text)
    try:
        when = email.utils.parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


class HttpClient:
    """项目内唯一的异步 HTTP 客户端实现。"""

    def __init__(
        self,
        config: SearchConfig,
        client: AsyncSession | None = None,
        *,
        url_guard: PublicUrlGuard | None = None,
    ):
        self.config = config
        self.client = client or AsyncSession(timeout=config.timeout)
        self._owns_client = client is None
        self._url_guard = url_guard or PublicUrlGuard()
        self._capacity = {
            "search": asyncio.Semaphore(config.max_concurrent_requests),
            "fetch": asyncio.Semaphore(config.max_concurrent_fetches),
        }

    async def arequest(
        self,
        method: str,
        url: str,
        *,
        params: Mapping | None = None,
        json: Mapping | None = None,
        headers: Mapping | None = None,
        timeout: float | None = None,
        request_kind: str = "generic",
    ) -> Response:
        last_error: Exception | None = None
        rate_limit_reset_ts: float | None = None
        for attempt in range(self.config.retry_attempts):
            retry_after: float | None = None
            try:
                # curl_cffi 的类型存根只接受受限的字面量集合；项目边界允许
                # 调用方继续使用通用的 HTTP 方法和 Mapping 类型。
                started = time.monotonic()
                semaphore = self._capacity.get(request_kind, self._capacity["fetch"])
                async with semaphore:
                    try:
                        response = await self._request_with_policy(
                            method=method,
                            url=url,
                            params=params,
                            json=json,
                            headers=headers,
                            timeout=timeout,
                            request_kind=request_kind,
                        )
                    except Exception:
                        await record_external_request(
                            category=request_kind,
                            status="failed",
                            duration_ms=round((time.monotonic() - started) * 1000),
                            detail={"attempt": attempt + 1},
                        )
                        raise
                await record_external_request(
                    category=request_kind,
                    status="completed" if response.status_code < 400 else "failed",
                    duration_ms=round((time.monotonic() - started) * 1000),
                    detail={"attempt": attempt + 1, "status_code": response.status_code},
                )
                if response.status_code in _RETRYABLE_STATUS:
                    # 服务端在 429/503 里的等待指示优先于本地指数曲线，
                    # 只受独立的 retry_after_max_seconds 上限约束。
                    retry_after = parse_retry_after(response.headers.get("Retry-After"))
                    if response.status_code in _RATE_LIMITED_STATUS:
                        window = min(
                            self.config.retry_after_max_seconds,
                            retry_after
                            if retry_after is not None
                            else _DEFAULT_RATE_LIMIT_WINDOW_SECONDS,
                        )
                        rate_limit_reset_ts = time.monotonic() + window
                    last_error = ToolRequestError(f"HTTP {response.status_code} from {url}")
                elif response.status_code >= 400:
                    if request_kind == "search" and response.status_code in _SEARCH_PROVIDER_FATAL:
                        # 搜索提供方账户级不可用：不重试、抛专用类型，让上层开熔断并快速收尾。
                        raise ProviderExhaustedError(
                            _SEARCH_PROVIDER_FATAL[response.status_code],
                            f"HTTP {response.status_code} from {url}",
                        )
                    raise ToolRequestError(
                        f"HTTP {response.status_code} from {url}", retryable=False
                    )
                else:
                    return response
            except (Timeout, RequestException) as exc:
                last_error = exc

            if attempt + 1 < self.config.retry_attempts:
                if retry_after is not None:
                    delay = min(self.config.retry_after_max_seconds, retry_after)
                else:
                    delay = min(
                        self.config.retry_max_seconds,
                        self.config.retry_initial_seconds * (2**attempt),
                    )
                # 打散并发请求的同步重试脉冲，避免互相踩着限流窗口反复撞。
                await asyncio.sleep(delay + random.uniform(0, delay * _JITTER_RATIO))

        error = ToolRequestError(
            f"请求失败（重试 {self.config.retry_attempts} 次）：{url}; reason={last_error}"
        )
        # 把最后一次观测到的限流恢复点附在错误上，供调用方做跨请求断路。
        error.rate_limit_reset_ts = rate_limit_reset_ts
        raise error from last_error

    async def _request_with_policy(
        self,
        *,
        method: str,
        url: str,
        params: Mapping | None,
        json: Mapping | None,
        headers: Mapping | None,
        timeout: float | None,
        request_kind: str,
    ) -> Response:
        if request_kind != "fetch":
            return await self._request_once(
                self.client,
                method=method,
                url=url,
                params=params,
                json=json,
                headers=headers,
                timeout=timeout,
                allow_redirects=True,
            )

        current_url = url
        current_method = method
        current_params = params
        current_json = json
        for redirect_count in range(_MAX_PUBLIC_REDIRECTS + 1):
            resolved = await self._url_guard.resolve(current_url)
            if self._owns_client:
                # A dedicated handle lets CURLOPT_RESOLVE pin this hop to the
                # addresses checked above. trust_env=False also prevents an
                # ambient proxy from bypassing the destination policy.
                async with AsyncSession(
                    timeout=self.config.timeout,
                    trust_env=False,
                    allow_redirects=False,
                    curl_options={CurlOpt.RESOLVE: resolved.curl_resolve},
                ) as guarded_client:
                    response = await self._request_once(
                        guarded_client,
                        method=current_method,
                        url=resolved.url,
                        params=current_params,
                        json=current_json,
                        headers=headers,
                        timeout=timeout,
                        allow_redirects=False,
                    )
            else:
                # Dependency-injected clients are used by deterministic tests.
                # They still receive normalized URLs and manual redirect policy.
                response = await self._request_once(
                    self.client,
                    method=current_method,
                    url=resolved.url,
                    params=current_params,
                    json=current_json,
                    headers=headers,
                    timeout=timeout,
                    allow_redirects=False,
                )

            primary_ip = str(getattr(response, "primary_ip", "") or "")
            if primary_ip:
                self._url_guard.ensure_public_ip(primary_ip)
                if primary_ip not in resolved.addresses:
                    raise ToolRequestError("连接目标与已校验的 DNS 地址不一致。", retryable=False)

            if response.status_code not in _REDIRECT_STATUS:
                return response
            location = response.headers.get("Location")
            if not location:
                return response
            if redirect_count >= _MAX_PUBLIC_REDIRECTS:
                raise ToolRequestError("来源重定向次数过多。", retryable=False)
            current_url = urljoin(resolved.url, location)
            current_params = None
            if response.status_code == 303 or (
                response.status_code in {301, 302} and current_method.upper() == "POST"
            ):
                current_method = "GET"
                current_json = None

        raise ToolRequestError("来源重定向次数过多。", retryable=False)

    @staticmethod
    async def _request_once(
        client: AsyncSession,
        *,
        method: str,
        url: str,
        params: Mapping | None,
        json: Mapping | None,
        headers: Mapping | None,
        timeout: float | None,
        allow_redirects: bool,
    ) -> Response:
        # curl_cffi's stubs expose narrower literal/collection types than this
        # project boundary intentionally accepts.
        request = cast(Any, client.request)
        return await request(
            method,
            url,
            params=params,
            json=json,
            headers=headers,
            timeout=timeout,
            allow_redirects=allow_redirects,
            impersonate=random.choice(_IMPERSONATE_TARGETS),
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.close()
