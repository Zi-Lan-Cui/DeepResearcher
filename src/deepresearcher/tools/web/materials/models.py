"""研究过程中可重建材料的存储模型。"""

import hashlib
from typing import Any

from pydantic import BaseModel, Field

from deepresearcher.tools.web.documents import DocumentOutlineItem, DocumentRef


class SearchResultSet(BaseModel):
    """一次 SearchSources 调用产生的完整候选集合。"""

    search_id: str
    run_id: str
    queries: list[str] = Field(default_factory=list)
    results: list[dict[str, Any]] = Field(default_factory=list)


class StoredDocument(BaseModel):
    """可按行复算的规范化正文；不直接暴露给 Agent。"""

    ref: DocumentRef
    content: str


def build_stored_document(
    *,
    text: str,
    title: str,
    source_url: str,
    published_at: str,
    retrieval_method: str,
    support_ceiling: str,
    token_count: int,
    outline: list[DocumentOutlineItem],
) -> StoredDocument:
    """由规范化正文确定性生成稳定句柄。"""

    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    identity = hashlib.sha256(f"{source_url}\0{text}".encode("utf-8")).hexdigest()
    ref = DocumentRef(
        document_id=f"doc-{identity}",
        title=title,
        source_url=source_url,
        published_at=published_at,
        retrieval_method=retrieval_method,
        support_ceiling=support_ceiling,
        content_hash=content_hash,
        line_count=len(text.splitlines()),
        token_count=token_count,
        outline=outline,
    )
    return StoredDocument(ref=ref, content=text)
