"""Writer 的内部准备态和校验结果。"""

from collections.abc import Callable
from dataclasses import dataclass

from deepresearcher.context.execution import AgentExecutionScope
from deepresearcher.evidence.models import Evidence
from deepresearcher.schemas import Citation, ParagraphBinding
from deepresearcher.schemas.sources import source_domain


def evidence_index_card(evidence: Evidence) -> dict[str, object]:
    """Writer 目录视图：只提供选材字段，不提前交付原文和完整 URL。"""
    return {
        "evidence_id": evidence.evidence_id,
        "claim": evidence.claim,
        "support": evidence.support,
        "source_title": evidence.source_title,
        "source_domain": source_domain(evidence.source_url),
        "source_type": evidence.source_profile.source_type,
        **({"published_at": evidence.published_at} if evidence.published_at else {}),
    }


def evidence_detail_card(evidence: Evidence) -> dict[str, object]:
    """ReadEvidence 详情视图：交付写作和引用所需的最小可验证材料。"""
    return {
        "evidence_id": evidence.evidence_id,
        "claim": evidence.claim,
        "quote": evidence.quote,
        "source_url": evidence.source_url,
        "source_title": evidence.source_title,
        "support": evidence.support,
        "source_type": evidence.source_profile.source_type,
        "authority_tier": evidence.source_profile.authority_tier,
        **({"published_at": evidence.published_at} if evidence.published_at else {}),
    }


@dataclass
class WriterLoopContext:
    """不进入 State 的 Writer 运行时依赖。"""

    scope: AgentExecutionScope
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
