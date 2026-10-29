"""内置网页抓取 Provider。"""

from typing import TYPE_CHECKING

from deepresearcher.tools.errors import ToolConfigurationError
from deepresearcher.tools.web.fetch.providers.aliyun import AliyunFetchProvider
from deepresearcher.tools.web.fetch.providers.direct import DirectHttpFetchProvider

if TYPE_CHECKING:
    from deepresearcher.config import SearchConfig
    from deepresearcher.tools.transport.aliyun import AliyunDtsApi
    from deepresearcher.tools.transport.http_client import HttpClient
    from deepresearcher.tools.web.fetch.protocol import FetchProvider

__all__ = ["AliyunFetchProvider", "DirectHttpFetchProvider", "build_fetch_providers"]


def build_fetch_providers(
    config: "SearchConfig",
    http_client: "HttpClient | None",
    aliyun_client: "AliyunDtsApi | None",
) -> list["FetchProvider"]:
    """按 config.fetch_provider_order 构建抓取器，初始化失败换下一个；全失败时报出每家原因。"""
    builders = {
        "aliyun": lambda: AliyunFetchProvider(config, _require_aliyun(aliyun_client)),
        "direct": lambda: DirectHttpFetchProvider(config, http_client),
    }
    providers: list["FetchProvider"] = []
    failures: list[str] = []
    for name in config.fetch_provider_order:
        builder = builders.get(name)
        if builder is None:
            failures.append(f"{name}：未实现的抓取器")
            continue
        try:
            providers.append(builder())
        except ToolConfigurationError as exc:
            failures.append(f"{name}：{exc}")
    if not providers:
        raise ToolConfigurationError("没有可用的抓取器 —— " + "；".join(failures))
    return providers


def _require_aliyun(client: "AliyunDtsApi | None") -> "AliyunDtsApi":
    if client is None:
        raise ToolConfigurationError("阿里云客户端未创建")
    return client
