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
            # 关键：Tavily advanced depth 对无法抽取正文的页面（付费墙/JS 页/多数站点）
            # 显式返回 JSON null。`item.get(k, "")` 只在键“缺失”时给默认值，键存在但为
            # null 时返回 None —— 会把 None 灌进 str 契约，令 SearchToolResult 的
            # Pydantic 校验整批失败（真实评测中被毒化到每次搜索，researcher 反复重试空转）。
            # `or ""` 同时折叠缺失与 null，且避免 str(None)=="None" 的字符串污染。
            return [
                {
                    "title": str(item.get("title") or ""),
                    "url": item["url"],
                    "snippet": str(item.get("content") or ""),
                    "raw_content": str(item.get("raw_content") or ""),
                    "content_provider": "tavily",
                    "score": float(item.get("score") or 0.0),
                    "published_at": str(item.get("published_date") or ""),
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
                    "title": str(item.get("title") or ""),
                    "url": item["link"],
                    "snippet": str(item.get("snippet") or ""),
                    "content_provider": "serpapi",
                    "score": float(limit - index) / limit,
                    "published_at": str(item.get("date") or ""),
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
                    "title": str(item.get("title") or ""),
                    "url": clean_url(str(item.get("url", ""))),
                    "snippet": str(item.get("content") or ""),
                    "content_provider": "baidu",
                    "score": float(limit - index) / limit,
                    "published_at": str(item.get("date") or ""),
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
