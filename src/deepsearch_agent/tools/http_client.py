"""异步 HTTP 客户端：统一处理浏览器 TLS 指纹、超时和重试。"""

import asyncio
import random
from collections.abc import Mapping
from typing import Any, cast

from curl_cffi.requests import AsyncSession, Response
from curl_cffi.requests.exceptions import RequestException, Timeout
from curl_cffi.requests.impersonate import DEFAULT_CHROME, DEFAULT_FIREFOX, DEFAULT_SAFARI

from deepsearch_agent.config import SearchConfig
from deepsearch_agent.tools.errors import ToolRequestError

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_IMPERSONATE_TARGETS = (DEFAULT_CHROME, DEFAULT_FIREFOX, DEFAULT_SAFARI)


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
        for attempt in range(self.config.retry_attempts):
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
                delay = min(
                    self.config.retry_max_seconds,
                    self.config.retry_initial_seconds * (2**attempt),
                )
                await asyncio.sleep(delay)

        raise ToolRequestError(
            f"请求失败（重试 {self.config.retry_attempts} 次）：{url}; reason={last_error}"
        ) from last_error

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.close()
