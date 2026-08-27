"""Writer 的内部准备态和校验结果。"""

from dataclasses import dataclass

from deepsearch_agent.evidence.models import Evidence
from deepsearch_agent.schemas import Citation, ParagraphBinding


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
