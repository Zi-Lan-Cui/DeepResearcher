"""网页搜索服务、Provider 和结果契约。"""

from deepresearcher.tools.web.search.client import SearchClient
from deepresearcher.tools.web.search.models import SearchCandidate, SearchResult, SearchToolResult
from deepresearcher.tools.web.search.protocol import SearchProvider
from deepresearcher.tools.web.search.providers import (
    AliyunSearchProvider,
    BaiduSearchProvider,
    SerpApiSearchProvider,
    TavilySearchProvider,
)
from deepresearcher.tools.web.search.service import SearchTool

__all__ = [
    "AliyunSearchProvider",
    "BaiduSearchProvider",
    "SearchCandidate",
    "SearchClient",
    "SearchProvider",
    "SearchResult",
    "SearchTool",
    "SearchToolResult",
    "SerpApiSearchProvider",
    "TavilySearchProvider",
]
