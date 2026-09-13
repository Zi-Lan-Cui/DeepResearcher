"""阿里云 DTS AI WebSearch Provider。"""

from deepresearcher.tools.errors import ToolParseError, ToolRequestError
from deepresearcher.tools.web.aliyun import AliyunDtsApi
from deepresearcher.tools.web.search.models import SearchResult


class AliyunSearchProvider:
    def __init__(self, client: AliyunDtsApi):
        self._client = client

    async def asearch(self, query: str, limit: int) -> list[SearchResult]:
        body = await self._client.web_search(query, limit)
        if not body or not bool(getattr(body, "success", False)):
            message = getattr(body, "error_message", "") if body else "空响应"
            raise ToolRequestError(f"阿里云 WebSearch 返回失败：{message or '未知错误'}")
        try:
            items = getattr(body, "search_result", None) or []
            return [
                {
                    "title": str(getattr(item, "title", "") or ""),
                    "url": str(getattr(item, "url", "") or ""),
                    "snippet": str(getattr(item, "snippet", "") or ""),
                    "content_provider": "aliyun",
                    "score": float(limit - index) / max(1, limit),
                    "published_at": "",
                }
                for index, item in enumerate(items[:limit])
                if getattr(item, "url", None)
            ]
        except (TypeError, ValueError) as exc:
            raise ToolParseError(f"阿里云 WebSearch 响应格式异常：{exc}") from exc
