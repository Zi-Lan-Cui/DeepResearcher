"""来源抓取与读取层的输入输出契约。"""

from typing import Literal

from pydantic import BaseModel, Field

from deepresearcher.evidence.models import Evidence
from deepresearcher.state import SubTask
from deepresearcher.tools.web.documents import DocumentView
from deepresearcher.tools.web.parsing.models import ParsedContent


class SourceDocument(ParsedContent, total=False):
    """解析正文及其来源、传输、缓存元数据。"""

    status: Literal["completed", "failed"]
    source_url: str
    final_url: str
    name: str
    ext: str
    content_type: str
    modality: str
    raw_bytes: int
    status_code: int
    content_hash: str
    error: str
    error_code: str
    retrieval_method: str
    support_ceiling: str
    published_at: str  # 由搜索结果携带的发布时间；reader 读取后附加，不进 L2 抓取缓存
    provider_request_id: str
    url_type: str
    fetch_duration_ms: float
    parse_duration_ms: float
    cache_hit: bool


class SourceReaderToolResult(BaseModel):
    """单来源读取结果；新路径返回文档，旧抽取基线仍可返回 Evidence。"""

    task_id: str
    status: Literal["completed", "failed", "skipped"]
    evidences: list[Evidence] = Field(default_factory=list)
    documents: list[DocumentView] = Field(default_factory=list)
    source_url: str = ""
    error: str = ""
    reason_code: str = ""


def failed_read(task: SubTask, error: Exception) -> SourceReaderToolResult:
    return SourceReaderToolResult(task_id=task["id"], status="failed", error=str(error)[:500])


def skipped_read(
    task: SubTask, *, source_url: str, reason_code: str, reason: str
) -> SourceReaderToolResult:
    return SourceReaderToolResult(
        task_id=task["id"],
        status="skipped",
        source_url=source_url,
        reason_code=reason_code,
        error=reason[:500],
    )
