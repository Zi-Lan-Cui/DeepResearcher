"""异步 HTTP 客户端：统一处理浏览器 TLS 指纹、超时和重试。"""

import asyncio
import email.utils
import random
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, cast

from curl_cffi.requests import AsyncSession, Response
from curl_cffi.requests.exceptions import RequestException, Timeout
from curl_cffi.requests.impersonate import DEFAULT_CHROME, DEFAULT_FIREFOX, DEFAULT_SAFARI

from deepsearch_agent.config import SearchConfig
from deepsearch_agent.tools.errors import ToolRequestError

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
# 429/503 是“配额/限流”型失败：除了退避重试，还要给调用方留下跨请求的恢复点。
_RATE_LIMITED_STATUS = {429, 503}
_DEFAULT_RATE_LIMIT_WINDOW_SECONDS = 30.0
_IMPERSONATE_TARGETS = (DEFAULT_CHROME, DEFAULT_FIREFOX, DEFAULT_SAFARI)
_JITTER_RATIO = 0.25


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

    def __init__(self, config: SearchConfig, client: AsyncSession | None = None):
        self.config = config
        self.client = client or AsyncSession(timeout=config.timeout)
        self._owns_client = client is None

    async def arequest(
        self,
        method: str,
        url: str,
        *,
        params: Mapping | None = None,
        json: Mapping | None = None,
        headers: Mapping | None = None,
        timeout: float | None = None,
    ) -> Response:
        last_error: Exception | None = None
        rate_limit_reset_ts: float | None = None
        for attempt in range(self.config.retry_attempts):
            retry_after: float | None = None
            try:
                # curl_cffi 的类型存根只接受受限的字面量集合；项目边界允许
                # 调用方继续使用通用的 HTTP 方法和 Mapping 类型。
                request = cast(Any, self.client.request)
                response = await request(
                    method,
                    url,
                    params=params,
                    json=json,
                    headers=headers,
                    timeout=timeout,
                    impersonate=random.choice(_IMPERSONATE_TARGETS),
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

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.close()
