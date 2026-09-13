"""搜索 Provider 的最小能力契约。"""

from typing import Protocol

from deepresearcher.tools.web.search.models import SearchResult


class SearchProvider(Protocol):
    async def asearch(self, query: str, limit: int) -> list[SearchResult]: ...
