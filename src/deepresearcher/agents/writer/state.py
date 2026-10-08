"""Writer 的内部准备态、校验结果与内层 agent 的状态通道。"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Any

from langchain.agents import AgentState

from deepresearcher.agents.channels import replace_channel, retain_if_absent
from deepresearcher.agents.middleware.concurrency import ToolExecutionGate
from deepresearcher.evidence.models import Evidence
from deepresearcher.observability.execution import AgentExecutionScope
from deepresearcher.schemas import Citation, ParagraphBinding
from deepresearcher.schemas.sources import source_domain
from deepresearcher.state import merge_unique


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


def _restore_draft(value: "ValidatedDraft | dict[str, Any] | None") -> "ValidatedDraft | None":
    """从通道现值恢复草稿:同 run 与 checkpoint serde 往返后都已是 ValidatedDraft 实例,dict 只是防御性兜底。"""
    if value is None or isinstance(value, ValidatedDraft):
        return value
    return ValidatedDraft(**value)


class WriterAgentState(AgentState):
    """ReportWriter 内层图的 state_schema:草稿循环业务状态的持久化载体。

    read_evidence_ids 走 merge_unique:并发 ReadEvidence 各交本批新增,按值折重。
    validated_draft、last_error、last_markdown 只由串行 CompleteReport 写,
    reducer 是 replace 或 retain。evidence_by_id 是 run 入口从外层 State 派生的
    只读目录,不进通道。
    """

    read_evidence_ids: Annotated[list[str], merge_unique]
    validated_draft: Annotated["ValidatedDraft | None", retain_if_absent]
    last_error: Annotated[str, replace_channel]
    last_markdown: Annotated[str, replace_channel]


@dataclass
class WriterLoopContext:
    """Writer 一次草稿循环的注入载荷:只读依赖留本对象,可变簿记进通道。"""

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
    tool_gate: ToolExecutionGate = field(default_factory=ToolExecutionGate)

    def initial_channel_state(self) -> dict[str, Any]:
        """构造 ainvoke 入口 input:当前簿记一次性种进内层通道。"""
        return {
            "read_evidence_ids": sorted(self.read_evidence_ids),
            "validated_draft": self.validated_draft,
            "last_error": self.last_error,
            "last_markdown": self.last_markdown,
        }

    def rebuild_loop_state(self, values: dict[str, Any]) -> None:
        """每次模型回合前从通道现值回填四项可变簿记;只读依赖不动。"""
        if "read_evidence_ids" in values:
            self.read_evidence_ids = set(values["read_evidence_ids"] or [])
        if "validated_draft" in values:
            self.validated_draft = _restore_draft(values["validated_draft"])
        if "last_error" in values:
            self.last_error = str(values["last_error"] or "")
        if "last_markdown" in values:
            self.last_markdown = str(values["last_markdown"] or "")


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
    # 模型为整篇回答拟的文章标题;内联兜底路径没有它,保持空串。
    # 放最后并带默认值:既有位置参数构造点不受影响。
    title: str = ""
