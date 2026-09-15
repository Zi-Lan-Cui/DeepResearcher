"""隔离阿里云生成式 SDK，避免云厂商类型泄漏到 Search/Fetch 服务。"""

import asyncio
import time
from typing import Any, Protocol

from deepresearcher.config import SearchConfig
from deepresearcher.observability.usage_runtime import record_external_request
from deepresearcher.tools.errors import ToolRequestError


class AliyunDtsApi(Protocol):
    """Provider 所需的最小阿里云能力；测试可直接注入 fake。"""

    async def web_search(self, query: str, limit: int) -> Any: ...

    async def web_fetch(self, url: str, output_format: str) -> Any: ...


class AliyunDtsClient:
    def __init__(self, config: SearchConfig, sdk_client: Any):
        self._config = config
        self._sdk_client = sdk_client

    async def web_search(self, query: str, limit: int) -> Any:
        # SDK 可能在 import 时固化默认凭据链的环境变量，因此延迟到
        # get_settings() 加载 env/.env 后再导入。
        from alibabacloud_dtsai20260401 import models as dts_models

        request = dts_models.WebSearchRequest(
            region_id=self._config.aliyun_region_id,
            query=query,
            max_results=min(max(1, limit), 50),
            agent_name=self._config.aliyun_agent_name,
        )
        return await self._call(
            "search",
            self._sdk_client.web_search_async(request),
        )

    async def web_fetch(self, url: str, output_format: str) -> Any:
        from alibabacloud_dtsai20260401 import models as dts_models

        request = dts_models.WebFetchRequest(
            region_id=self._config.aliyun_region_id,
            url=url,
            output_format=output_format,
            agent_name=self._config.aliyun_agent_name,
        )
        return await self._call(
            "fetch",
            self._sdk_client.web_fetch_async(request),
        )

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
            raise ToolRequestError(
                f"阿里云 Web{category.title()} 请求失败：{message[:500]}"
            ) from exc
        finally:
            await record_external_request(
                category=category,
                status=status,
                duration_ms=int((time.monotonic() - started) * 1000),
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
