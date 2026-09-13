"""网页抓取 Provider 的最小能力契约。"""

from typing import Protocol

from deepresearcher.tools.web.fetch.models import SourceDocument


class FetchProvider(Protocol):
    """取得并标准化一个 URL；不负责 Evidence 抽取。"""

    name: str

    async def afetch(
        self,
        url: str,
        *,
        fetch_timeout: float | None = None,
        parse_timeout: float | None = None,
    ) -> SourceDocument: ...
