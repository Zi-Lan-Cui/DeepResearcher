"""来源抓取与读取层的输入输出契约。"""

from typing import Literal

from pydantic import BaseModel, Field

from deepsearch_agent.evidence.models import Evidence
from deepsearch_agent.parsers.models import ParsedContent
from deepsearch_agent.state import SubTask


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
    fetch_duration_ms: float
    parse_duration_ms: float
    cache_hit: bool


class SourceReaderToolResult(BaseModel):
    """单来源读取与 Evidence 抽取工具的稳定返回契约。"""

    task_id: str
    status: Literal["completed", "failed", "skipped"]
    evidences: list[Evidence] = Field(default_factory=list)
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
