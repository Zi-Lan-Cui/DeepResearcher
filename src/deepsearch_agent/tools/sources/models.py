"""来源读取层的输入输出契约。"""

from typing import Literal

from pydantic import BaseModel, Field

from deepsearch_agent.evidence.models import Evidence
from deepsearch_agent.state import SubTask


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
