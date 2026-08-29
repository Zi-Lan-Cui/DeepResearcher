"""Writer 的内部准备态和校验结果。"""

from collections.abc import Callable
from dataclasses import dataclass

from deepsearch_agent.evidence.models import Evidence
from deepsearch_agent.schemas import Citation, ParagraphBinding


@dataclass
class WriterRuntimeContext:
    """不进入 State 的 Writer 运行时依赖。"""

    evidence_by_id: dict[str, Evidence]
    read_evidence_ids: set[str]
    read_batch_size: int
    max_markdown_chars: int
    emit: Callable[[str, dict[str, object]], None]
    validated_draft: "ValidatedDraft | None" = None
    last_error: str = ""
    last_markdown: str = ""
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
