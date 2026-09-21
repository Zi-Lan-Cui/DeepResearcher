"""SerpAPI 搜索 SearchProvider。"""

from deepresearcher.config import SearchConfig
from deepresearcher.tools.errors import ToolParseError, ToolRequestError
from deepresearcher.tools.transport.http_client import HttpClient
from deepresearcher.tools.web.search.models import SearchResult


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
