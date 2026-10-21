"""Writer 子图的中间产物、校验结果，以及内层 agent 的状态通道定义。"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any

from langchain.agents import AgentState

from deepresearcher.agents.channels import replace_channel, retain_if_absent
from deepresearcher.evidence.models import Evidence
from deepresearcher.observability.execution import AgentExecutionScope
from deepresearcher.schemas import Citation, ParagraphBinding
from deepresearcher.schemas.sources import source_domain
from deepresearcher.state import merge_unique


def evidence_index_card(evidence: Evidence) -> dict[str, object]:
    """返回证据的目录视图，只含选材字段；正文与完整 URL 须经 ReadEvidence 获取。"""
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
    """返回证据的详情视图，在目录字段之外补 quote、来源 URL 与权威等级。"""
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
    """把通道中的草稿值统一为 ValidatedDraft 实例。"""
    if value is None or isinstance(value, ValidatedDraft):
        return value
    # checkpoint 反序列化后可能得到 dict，需在此转回实例。
    return ValidatedDraft(**value)


class WriterAgentState(AgentState):
    """ReportWriter 内层图的状态定义，承载草稿循环的业务状态。

    read_evidence_ids 由并发 ReadEvidence 各写本批新增，merge_unique 去重；
    validated_draft、last_error、last_markdown 只由串行 CompleteReport 写，
    用 replace_channel 整体覆盖。evidence_by_id 从外层 State 派生，只读、不设通道。
    """

    read_evidence_ids: Annotated[list[str], merge_unique]
    validated_draft: Annotated["ValidatedDraft | None", retain_if_absent]
    last_error: Annotated[str, replace_channel]
    last_markdown: Annotated[str, replace_channel]


@dataclass
class WriterLoopContext:
    """Writer 单次草稿循环的只读依赖与可变状态载体。

    可变状态经 LangGraph 通道传递，每轮模型调用前由 rebuild_loop_state 回填。
    """

    scope: AgentExecutionScope
    evidence_by_id: dict[str, Evidence]
    read_evidence_ids: set[str]
    read_batch_size: int
    max_markdown_chars: int
    emit: Callable[[str, dict[str, object]], None]
    validated_draft: "ValidatedDraft | None" = None
    last_error: str = ""
    last_markdown: str = ""

    def initial_channel_state(self) -> dict[str, Any]:
        """构造 ainvoke 的入口输入，把当前状态一次性写入内层通道。"""
        return {
            "read_evidence_ids": sorted(self.read_evidence_ids),
            "validated_draft": self.validated_draft,
            "last_error": self.last_error,
            "last_markdown": self.last_markdown,
        }

    def rebuild_loop_state(self, values: dict[str, Any]) -> None:
        """用通道现值更新四项可变状态，只读依赖保持不变。"""
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
    # 模型为整篇回答生成的文章标题。内联回退路径不产出标题，保持空串。
    # 置于末位并带默认值，已有按位置传参的调用点无需改动。
    title: str = ""
