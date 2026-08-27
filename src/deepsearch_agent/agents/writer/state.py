"""Writer 的内部准备态和校验结果。"""

from dataclasses import dataclass
from typing import TypedDict

from langchain_core.messages import BaseMessage

from deepsearch_agent.evidence.models import Evidence
from deepsearch_agent.schemas import Citation, ParagraphBinding


class WriterState(TypedDict, total=False):
    """Writer 子图可保存的业务状态。"""

    messages: list[BaseMessage]
    evidence_catalogue: str
    read_evidence_ids: list[str]
    draft: str
    validation_error: str
    status: str
    attempts: int


@dataclass(frozen=True)
class WriterRuntimeContext:
    """不进入 State 的 Writer 运行时依赖。"""

    evidence_by_id: dict[str, Evidence]
    artifact_max_text_chars: int = 1_000


@dataclass(frozen=True)
class PreparedEvidence:
    by_id: dict[str, Evidence]
    catalogue: str


@dataclass(frozen=True)
class ValidatedDraft:
    body: str
    paragraph_bindings: list[ParagraphBinding]
    citations: list[Citation]
    selected_evidence_ids: list[str]


@dataclass(frozen=True)
class GenerationResult:
    draft: ValidatedDraft | None
    last_markdown: str
    validation_error: str
