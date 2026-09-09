"""搜索供应商适配器；只在这里处理各服务的请求和响应格式。"""

import re
from typing import Protocol

from deepsearch_agent.config import SearchConfig
from deepsearch_agent.tools.errors import ToolParseError, ToolRequestError
from deepsearch_agent.tools.search.models import SearchResult
from deepsearch_agent.tools.transport.http_client import HttpClient


class SearchProvider(Protocol):
    async def asearch(self, query: str, limit: int) -> list[SearchResult]: ...


class TavilySearchProvider:
    def __init__(self, config: SearchConfig, http: HttpClient):
        self.config, self.http = config, http

    async def asearch(self, query: str, limit: int) -> list[SearchResult]:
        response = await self.http.arequest(
            "POST",
            "https://api.tavily.com/search",
            json={
                "api_key": self.config.tavily_api_key,
                "query": query,
                "search_depth": "advanced",
                "max_results": limit,
                "include_answer": False,
                "include_raw_content": self.config.tavily_include_raw_content,
            },
            request_kind="search",
        )
        try:
            return [
                {
                    "title": item.get("title", ""),
                    "url": item["url"],
                    "snippet": item.get("content", ""),
                    "raw_content": item.get("raw_content", ""),
                    "content_provider": "tavily",
                    "score": float(item.get("score", 0.0)),
                    "published_at": str(item.get("published_date", "")),
                }
                for item in response.json().get("results", [])
                if item.get("url")
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise ToolParseError(f"Tavily 响应格式异常：{exc}") from exc


class SerpApiSearchProvider:
    def __init__(self, config: SearchConfig, http: HttpClient):
        self.config, self.http = config, http

    async def asearch(self, query: str, limit: int) -> list[SearchResult]:
        response = await self.http.arequest(
            "GET",
            "https://serpapi.com/search.json",
            params={
                "engine": "google",
                "q": query,
                "api_key": self.config.serpapi_api_key,
                "num": limit,
            },
            request_kind="search",
        )
        try:
            data = response.json()
            if data.get("error"):
                raise ToolRequestError(
                    f"SerpAPI 返回错误：{str(data['error'])[:500]}", retryable=False
                )
            return [
                {
                    "title": item.get("title", ""),
                    "url": item["link"],
                    "snippet": item.get("snippet", ""),
                    "content_provider": "serpapi",
                    "score": float(limit - index) / limit,
                    "published_at": str(item.get("date", "")),
                }
                for index, item in enumerate(data.get("organic_results", []))
                if item.get("link")
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise ToolParseError(f"SerpAPI 响应格式异常：{exc}") from exc


class BaiduSearchProvider:
    """百度千帆 AI Search v2 适配器。"""

    endpoint = "https://qianfan.baidubce.com/v2/ai_search/web_search"

    def __init__(self, config: SearchConfig, http: HttpClient):
        self.config, self.http = config, http

    async def asearch(self, query: str, limit: int) -> list[SearchResult]:
        response = await self.http.arequest(
            "POST",
            self.endpoint,
            headers={
                "X-Appbuilder-Authorization": f"Bearer {self.config.baidu_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "messages": [{"content": query, "role": "user"}],
                "search_source": "baidu_search_v2",
                "resource_type_filter": [{"type": "web", "top_k": min(limit, 50)}],
            },
            request_kind="search",
        )
        try:
            data = response.json()
            if data.get("code") or (data.get("message") and "references" not in data):
                detail = data.get("message") or f"code={data.get('code')}"
                raise ToolRequestError(f"百度搜索返回错误：{str(detail)[:500]}", retryable=False)
            references = data.get("references", [])
            if not isinstance(references, list):
                raise TypeError("references 不是列表")
            return [
                {
                    "title": str(item.get("title", "")),
                    "url": clean_url(str(item.get("url", ""))),
                    "snippet": str(item.get("content", "")),
                    "content_provider": "baidu",
                    "score": float(limit - index) / limit,
                    "published_at": str(item.get("date", "")),
                }
                for index, item in enumerate(references[:limit])
                if clean_url(str(item.get("url", ""))) and item.get("type", "web") == "web"
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise ToolParseError(f"百度搜索响应格式异常：{exc}") from exc


_MARKDOWN_URL = re.compile(r"^\[[^]]*\]\((https?://[^)]+)\)$")


def clean_url(url: str) -> str:
    """兼容百度响应中可能出现的 Markdown 链接形式。"""
    value = url.strip()
    match = _MARKDOWN_URL.match(value)
    return match.group(1) if match else value
