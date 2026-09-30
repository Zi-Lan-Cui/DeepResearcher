"""隔离阿里云生成式 SDK，避免云厂商类型泄漏到 Search/Fetch 服务。

SDK 响应在本边界一次性转换为窄 dataclass:success 门、错误分类与字段读取都收拢于此,
两侧 provider 只见 typed 字段,不再 getattr 探测云厂商响应形状。
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from deepresearcher.config import SearchConfig
from deepresearcher.errors import clip_text
from deepresearcher.timing import elapsed_ms
from deepresearcher.tools.errors import ProviderExhaustedError, ToolRequestError
from deepresearcher.usage_runtime import record_external_request

# 阿里云 SDK 账户级错误码 → 熔断 user_code(与 http_client 的 _SEARCH_PROVIDER_FATAL 同词表)。
# 只映射稳定的鉴权/授权/额度码:瞬态 Throttling 一律留给 retryable 的 ToolRequestError——
# 误熔断会停掉一个健康提供方,比多撞一次网络代价大得多。
_ALIYUN_ACCOUNT_FATAL: dict[str, str] = {
    "InvalidAccessKeyId": "invalid_key",
    "InvalidSecurityToken": "invalid_key",
    "MissingSecurityToken": "invalid_key",
    "SignatureDoesNotMatch": "invalid_key",
    "IncompleteSignature": "invalid_key",
    "Forbidden": "forbidden",
    "Forbidden.RAM": "forbidden",
    "NoPermission": "forbidden",
    "AccessDenied": "forbidden",
    "QuotaExhausted": "quota_exhausted",
    "InsufficientBalance": "insufficient_credit",
}


@dataclass(frozen=True)
class AliyunWebSearchItem:
    url: str
    title: str = ""
    snippet: str = ""


@dataclass(frozen=True)
class AliyunWebSearchBody:
    items: list[AliyunWebSearchItem] = field(default_factory=list)


@dataclass(frozen=True)
class AliyunWebFetchBody:
    content: str
    title: str = ""
    content_format: str = ""
    url: str = ""
    http_status_code: int = 200
    request_id: str = ""
    url_type: str = ""


class AliyunDtsApi(Protocol):
    """Provider 所需的最小阿里云能力；测试可直接注入 fake。"""

    async def web_search(self, query: str, limit: int) -> AliyunWebSearchBody: ...

    async def web_fetch(self, url: str, output_format: str) -> AliyunWebFetchBody: ...


class AliyunDtsClient:
    def __init__(self, config: SearchConfig, sdk_client: Any):
        self._config = config
        self._sdk_client = sdk_client

    async def web_search(self, query: str, limit: int) -> AliyunWebSearchBody:
        # SDK 可能在 import 时固化默认凭据链的环境变量，因此延迟到
        # get_settings() 加载 env/.env 后再导入。
        from alibabacloud_dtsai20260401 import models as dts_models

        request = dts_models.WebSearchRequest(
            region_id=self._config.aliyun_region_id,
            query=query,
            max_results=min(max(1, limit), 50),
            agent_name=self._config.aliyun_agent_name,
        )
        body = await self._call("search", self._sdk_client.web_search_async(request))
        self._require_success("search", body)
        return AliyunWebSearchBody(
            items=[
                AliyunWebSearchItem(
                    url=str(getattr(item, "url", "") or ""),
                    title=str(getattr(item, "title", "") or ""),
                    snippet=str(getattr(item, "snippet", "") or ""),
                )
                for item in (getattr(body, "search_result", None) or [])
            ]
        )

    async def web_fetch(self, url: str, output_format: str) -> AliyunWebFetchBody:
        from alibabacloud_dtsai20260401 import models as dts_models

        request = dts_models.WebFetchRequest(
            region_id=self._config.aliyun_region_id,
            url=url,
            output_format=output_format,
            agent_name=self._config.aliyun_agent_name,
        )
        body = await self._call("fetch", self._sdk_client.web_fetch_async(request))
        self._require_success("fetch", body)
        return AliyunWebFetchBody(
            content=str(getattr(body, "content", "") or ""),
            title=str(getattr(body, "title", "") or ""),
            content_format=str(getattr(body, "content_format", "") or "").lower(),
            url=str(getattr(body, "url", "") or ""),
            http_status_code=int(getattr(body, "http_status_code", 200) or 200),
            request_id=str(getattr(body, "request_id", "") or ""),
            url_type=str(getattr(body, "url_type", "") or ""),
        )

    @staticmethod
    def _require_success(category: str, body: Any) -> None:
        """业务级失败门(SDK 未抛异常但 success=false):两家 provider 共用同一文案与分类。"""
        if body is None or not bool(getattr(body, "success", False)):
            message = str(getattr(body, "error_message", "") or "") if body is not None else "空响应"
            raise ToolRequestError(f"阿里云 Web{category.title()} 返回失败：{message or '未知错误'}")

    async def _call(self, category: str, request: Any) -> Any:
        started = time.monotonic()
        status = "success"
        try:
            response = await asyncio.wait_for(request, timeout=self._config.timeout)
            return response.body
        except asyncio.TimeoutError as exc:
            status = "timeout"
            raise ToolRequestError(f"阿里云 Web{category.title()} 请求超时。") from exc
        except Exception as exc:
            status = "failed"
            message = getattr(exc, "message", None) or str(exc)
            code = str(getattr(exc, "code", "") or "")
            user_code = _ALIYUN_ACCOUNT_FATAL.get(code)
            if user_code is not None:
                # 账户级失败抛专用类型:让 SearchService 开熔断、跨 worker 快速收尾,
                # 而非每个请求各自反复撞同一个鉴权错误。
                raise ProviderExhaustedError(
                    user_code,
                    f"阿里云 Web{category.title()} 账户级失败（{code}）：{clip_text(message)}",
                ) from exc
            raise ToolRequestError(
                f"阿里云 Web{category.title()} 请求失败：{clip_text(message)}"
            ) from exc
        finally:
            await record_external_request(
                category=category,
                status=status,
                duration_ms=elapsed_ms(started),
                detail={"provider": "aliyun"},
            )


def create_aliyun_dts_client(config: SearchConfig) -> AliyunDtsClient:
    """使用官方默认凭据链创建客户端，不在应用配置中持有 AccessKey。

    云厂商 SDK 必须在 ``get_settings()`` 完成 dotenv 加载后导入；否则
    其默认凭据链会把导入当时的空环境变量缓存到进程内。
    """
    from alibabacloud_credentials.client import Client as CredentialClient
    from alibabacloud_dtsai20260401.client import Client as DtsClient
    from alibabacloud_tea_openapi import models as open_api_models

    credential = CredentialClient()
    sdk_config = open_api_models.Config(credential=credential)
    sdk_config.endpoint = config.aliyun_endpoint
    return AliyunDtsClient(config, DtsClient(sdk_config))
