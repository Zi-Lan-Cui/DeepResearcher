"""Tavily 搜索适配器。"""

from deepresearcher.config import SearchConfig
from deepresearcher.tools.errors import ToolParseError
from deepresearcher.tools.transport.http_client import HttpClient
from deepresearcher.tools.web.search.models import SearchResult


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
            # Tavily 对无法抽取正文的页面会显式返回 null，统一折叠为空字符串，
            # 防止单条异常值破坏整批 SearchToolResult 校验。
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
