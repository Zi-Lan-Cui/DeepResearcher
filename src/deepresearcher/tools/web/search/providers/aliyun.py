"""阿里云 DTS AI WebSearch Provider。"""

from deepresearcher.tools.transport.aliyun import AliyunDtsApi
from deepresearcher.tools.web.search.models import SearchResult


class AliyunSearchProvider:
    def __init__(self, client: AliyunDtsApi):
        self._client = client

    async def asearch(self, query: str, limit: int) -> list[SearchResult]:
        body = await self._client.web_search(query, limit)
        return [
            {
                "title": item.title,
                "url": item.url,
                "snippet": item.snippet,
                "content_provider": "aliyun",
                "score": float(limit - index) / max(1, limit),
                "published_at": "",
            }
            for index, item in enumerate(body.items[:limit])
            if item.url
        ]
