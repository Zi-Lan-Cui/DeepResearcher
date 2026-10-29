"""内置搜索 Provider。"""

from typing import TYPE_CHECKING, Any

from deepresearcher.tools.errors import ToolConfigurationError
from deepresearcher.tools.web.search.providers.aliyun import AliyunSearchProvider
from deepresearcher.tools.web.search.providers.baidu import BaiduSearchProvider
from deepresearcher.tools.web.search.providers.deepseek import DeepSeekSearchProvider
from deepresearcher.tools.web.search.providers.serpapi import SerpApiSearchProvider
from deepresearcher.tools.web.search.providers.tavily import TavilySearchProvider

if TYPE_CHECKING:
    from deepresearcher.config import SearchConfig
    from deepresearcher.tools.transport.aliyun import AliyunDtsApi
    from deepresearcher.tools.transport.http_client import HttpClient

__all__ = [
    "AliyunSearchProvider",
    "BaiduSearchProvider",
    "DeepSeekSearchProvider",
    "SerpApiSearchProvider",
    "TavilySearchProvider",
    "build_search_providers",
]


def build_search_providers(
    config: "SearchConfig",
    http_client: "HttpClient",
    aliyun_client: "AliyunDtsApi | None",
) -> list[tuple[str, Any]]:
    """按 config.candidates 顺序构建供应商，初始化失败换下一个；全失败时报出每家原因。"""
    builders = {
        "aliyun": lambda: AliyunSearchProvider(_require_aliyun(aliyun_client)),
        "baidu": lambda: BaiduSearchProvider(config, http_client),
        "tavily": lambda: TavilySearchProvider(config, http_client),
        "serpapi": lambda: SerpApiSearchProvider(config, http_client),
        "deepseek": lambda: DeepSeekSearchProvider(config, http_client),
    }
    providers: list[tuple[str, Any]] = []
    failures: list[str] = []
    for name in config.candidates:
        builder = builders.get(name)
        if builder is None:
            failures.append(f"{name}：未实现的搜索供应商")
            continue
        if not config.ready(name):
            failures.append(f"{name}：未配置凭据")
            continue
        try:
            providers.append((name, builder()))
        except ToolConfigurationError as exc:
            failures.append(f"{name}：{exc}")
    if not providers:
        raise ToolConfigurationError("没有可用的搜索供应商 —— " + "；".join(failures))
    return providers


def _require_aliyun(client: "AliyunDtsApi | None") -> "AliyunDtsApi":
    if client is None:
        raise ToolConfigurationError("阿里云客户端未创建")
    return client
