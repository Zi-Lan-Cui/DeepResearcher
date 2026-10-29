"""DeepSeek 网页搜索 SearchProvider。"""

from deepresearcher.config import SearchConfig
from deepresearcher.tools.errors import ToolParseError
from deepresearcher.tools.transport.http_client import HttpClient
from deepresearcher.tools.web.search.models import SearchResult

# 接口对 count 的硬区间；越界是 400，不会自动截断。
_COUNT_MIN = 1
_COUNT_MAX = 50


class DeepSeekSearchProvider:
    """自定义检索引擎；一次请求可同时取回摘要与正文。"""

    name = "deepseek"

    def __init__(self, config: SearchConfig, http: HttpClient):
        self.config, self.http = config, http

    async def asearch(self, query: str, limit: int) -> list[SearchResult]:
        response = await self.http.arequest(
            "POST",
            f"{self.config.deepseek_base_url}/v1/tools/search",
            headers={"Authorization": f"Bearer {self.config.deepseek_api_key}"},
            json={
                "query": query,
                "count": min(_COUNT_MAX, max(_COUNT_MIN, limit)),
                "need_content": self.config.deepseek_search_content,
            },
            request_kind="search",
        )
        try:
            # summary 是给模型读的长摘要，snippet 只有一两百字；候选卡要进提示词，取长的那个。
            return [
                {
                    "title": str(item.get("title") or ""),
                    "url": item["url"],
                    "snippet": str(item.get("summary") or item.get("snippet") or ""),
                    "raw_content": str(item.get("content") or ""),
                    "content_provider": "deepseek",
                    "score": float(item.get("rerank_score") or 0.0),
                    "published_at": str(item.get("date") or ""),
                }
                for item in response.json().get("results") or []
                if item.get("url")
            ]
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ToolParseError(f"DeepSeek 搜索响应格式异常：{exc}") from exc
