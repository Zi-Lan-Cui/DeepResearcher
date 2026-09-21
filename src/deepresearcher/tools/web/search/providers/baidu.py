"""百度千帆 AI Search v2 SearchProvider。"""

import re

from deepresearcher.config import SearchConfig
from deepresearcher.tools.errors import ToolParseError, ToolRequestError
from deepresearcher.tools.transport.http_client import HttpClient
from deepresearcher.tools.web.search.models import SearchResult

_MARKDOWN_URL = re.compile(r"^\[[^]]*\]\((https?://[^)]+)\)$")


def clean_url(url: str) -> str:
    """兼容百度响应中可能出现的 Markdown 链接形式。"""
    value = url.strip()
    match = _MARKDOWN_URL.match(value)
    return match.group(1) if match else value


class BaiduSearchProvider:
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
