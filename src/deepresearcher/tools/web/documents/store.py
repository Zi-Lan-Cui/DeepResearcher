"""DocumentStore 协议。"""

from typing import Protocol

from deepresearcher.tools.web.documents.models import (
    DocumentGrepMatch,
    DocumentOutlineItem,
    DocumentReadRange,
    DocumentRef,
)


class DocumentStore(Protocol):
    """正文存储抽象；Agent 只能使用 document_id，不能接触物理路径。"""

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
    ) -> DocumentRef: ...

    async def get(self, document_id: str) -> DocumentRef: ...

    async def text(self, document_id: str) -> str: ...

    async def read(
        self,
        document_id: str,
        ranges: list[tuple[int, int]],
        *,
        max_lines: int,
        max_chars: int,
    ) -> list[DocumentReadRange]: ...

    async def grep(
        self,
        document_id: str,
        queries: list[str],
        *,
        context_lines: int,
        max_matches: int,
        max_chars: int,
    ) -> list[DocumentGrepMatch]: ...
