"""进程内研究材料存储：用于测试和 Redis 降级。"""

from deepresearcher.tools.web.documents import (
    DocumentOutlineItem,
    DocumentReadRange,
    DocumentRef,
    GrepResult,
)
from deepresearcher.tools.web.materials.models import (
    SearchResultSet,
    StoredDocument,
    build_stored_document,
)
from deepresearcher.tools.web.materials.store import grep_lines, read_lines


class MemoryResearchMaterialStore:
    """遵循同一协议的最小内存实现。"""

    def __init__(self) -> None:
        self._searches: dict[tuple[str, str], SearchResultSet] = {}
        self._documents: dict[str, StoredDocument] = {}
        self._fetch_index: dict[str, str] = {}

    async def put_search_results(self, result_set: SearchResultSet) -> None:
        self._searches[(result_set.run_id, result_set.search_id)] = result_set.model_copy(deep=True)

    async def get_search_results(self, run_id: str, search_id: str) -> SearchResultSet:
        try:
            return self._searches[(run_id, search_id)].model_copy(deep=True)
        except KeyError as exc:
            raise FileNotFoundError(f"搜索结果不存在：{search_id}") from exc

    async def put(
        self,
        *,
        text: str,
        title: str,
        source_url: str,
        published_at: str = "",
        retrieval_method: str = "origin_fetch",
        support_ceiling: str = "direct",
        token_count: int = 0,
        outline: list[DocumentOutlineItem] | None = None,
        fetch_key: str = "",
    ) -> DocumentRef:
        document = build_stored_document(
            text=text,
            title=title,
            source_url=source_url,
            published_at=published_at,
            retrieval_method=retrieval_method,
            support_ceiling=support_ceiling,
            token_count=token_count,
            outline=outline or [],
        )
        ref = document.ref
        self._documents[ref.document_id] = document
        if fetch_key:
            self._fetch_index[fetch_key] = ref.document_id
        return ref

    async def resolve_fetch(self, fetch_key: str) -> DocumentRef | None:
        document_id = self._fetch_index.get(fetch_key)
        document = self._documents.get(document_id or "")
        return document.ref.model_copy(deep=True) if document is not None else None

    async def get(self, document_id: str) -> DocumentRef:
        return (await self._document(document_id)).ref.model_copy(deep=True)

    async def text(self, document_id: str) -> str:
        return (await self._document(document_id)).content

    async def read(
        self,
        document_id: str,
        ranges: list[tuple[int, int]],
        *,
        max_lines: int,
        max_chars: int,
    ) -> list[DocumentReadRange]:
        document = await self._document(document_id)
        return read_lines(
            document.content.splitlines(),
            ranges,
            max_lines=max_lines,
            max_chars=max_chars,
        )

    async def grep(
        self,
        document_id: str,
        query: str,
        *,
        context_lines: int,
        max_matches: int,
        max_chars: int,
        offset: int = 0,
    ) -> GrepResult:
        document = await self._document(document_id)
        return grep_lines(
            document.content.splitlines(),
            query,
            context_lines=context_lines,
            max_matches=max_matches,
            max_chars=max_chars,
            offset=offset,
        )

    async def close(self) -> None:
        return None

    async def _document(self, document_id: str) -> StoredDocument:
        try:
            return self._documents[document_id]
        except KeyError as exc:
            raise FileNotFoundError(f"文档不存在：{document_id}") from exc
