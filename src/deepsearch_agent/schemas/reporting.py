"""报告域契约：写作任务书、引用元数据与段落绑定。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from deepsearch_agent.schemas.sources import SourceProfile


class CoveredTopic(BaseModel):
    """报告中必须处理的研究主题及其论证角色。"""

    topic: str
    role: str
    reason: str
    required: bool = True
    evidence_ids: list[str] = Field(default_factory=list, max_length=30)


class ReportBrief(BaseModel):
    """Supervisor 交给 Writer 的任务书；不携带 Evidence 正文。"""

    answer_goal: str
    covered_topics: list[CoveredTopic] = Field(min_length=1, max_length=6)
    required_points: list[str] = Field(default_factory=list, max_length=8)
    caveats: list[str] = Field(default_factory=list, max_length=6)


class ResearchAspect(BaseModel):
    """Supervisor 跨研究方向建立的证据支撑认知单元。

    它用于覆盖、冲突和缺口管理；不是 Researcher 任务方向，
    也不在概念上等于最终报告章节。
    """

    aspect_id: str = Field(min_length=1, max_length=80)
    topic: str = Field(min_length=1, max_length=500)
    role: str = Field(min_length=1, max_length=500)
    required: bool = True
    status: Literal["covered", "partial", "uncovered", "conflicted"]
    summary: str = Field(default="", max_length=2_000)
    evidence_ids: list[str] = Field(default_factory=list, max_length=30)
    remaining_gap: str = Field(default="", max_length=1_000)

    @model_validator(mode="after")
    def validate_grounding(self) -> "ResearchAspect":
        if self.status == "covered" and not self.evidence_ids:
            raise ValueError("covered 研究方面必须绑定至少一条 Evidence。")
        if self.status == "uncovered" and self.evidence_ids:
            raise ValueError("uncovered 研究方面不能绑定 Evidence。")
        if self.status in {"partial", "uncovered", "conflicted"} and not self.remaining_gap:
            raise ValueError(f"{self.status} 研究方面必须明确 remaining_gap。")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("同一研究方面不能重复绑定 Evidence。")
        return self


class ResearchSynthesis(BaseModel):
    """Supervisor 持续修订、最终冻结后交给 Writer 的研究综合稿。"""

    revision: int = Field(ge=1)
    based_on_working_set_revision: int = Field(ge=0)
    answer_goal: str = Field(min_length=1, max_length=2_000)
    overall_summary: str = Field(min_length=1, max_length=4_000)
    aspects: list[ResearchAspect] = Field(min_length=1, max_length=6)
    selected_evidence_ids: list[str] = Field(default_factory=list, max_length=50)
    open_gaps: list[str] = Field(default_factory=list, max_length=12)
    conflicts: list[str] = Field(default_factory=list, max_length=8)
    next_actions: list[str] = Field(default_factory=list, max_length=8)
    readiness: Literal["not_ready", "partial_ready", "complete_candidate"]
    decision_rationale: str = Field(min_length=1, max_length=2_000)

    @model_validator(mode="after")
    def validate_selection(self) -> "ResearchSynthesis":
        aspect_ids = [item.aspect_id for item in self.aspects]
        if len(set(aspect_ids)) != len(aspect_ids):
            raise ValueError("研究综合稿不能包含重复 aspect_id。")
        selected = list(
            dict.fromkeys(
                evidence_id
                for aspect in self.aspects
                for evidence_id in aspect.evidence_ids
            )
        )
        if len(selected) > 50:
            raise ValueError("研究综合稿最多选择 50 条 Evidence。")
        self.selected_evidence_ids = selected
        if self.readiness != "not_ready" and not self.selected_evidence_ids:
            raise ValueError("可交付研究综合稿必须选择至少一条 Evidence。")
        return self


class WriterDirective(BaseModel):
    """Supervisor 交给 Writer 的唯一写作交接契约。"""

    query: str
    report_brief: ReportBrief
    research_status: Literal["not_started", "running", "completed", "incomplete", "failed"]
    generation_mode: Literal["not_ready", "partial", "full"]
    evidence_ids: list[str] | None = None
    known_gaps: list[str] = Field(default_factory=list, max_length=8)
    revision_instructions: list[str] = Field(default_factory=list, max_length=8)
    previous_draft: str = ""


class Citation(BaseModel):
    """渲染后正文引用所需的可审计来源元数据。"""

    id: str
    url: str = ""
    title: str = ""
    quote: str = ""
    claim: str = ""
    source_profile: SourceProfile = Field(default_factory=SourceProfile)


class ParagraphBinding(BaseModel):
    """一块报告正文与其 Evidence 引用的绑定关系。"""

    text: str
    kind: Literal["evidence", "synthesis", "transition"]
    evidence_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_evidence_binding(self) -> "ParagraphBinding":
        if self.kind == "evidence" and not self.evidence_ids:
            raise ValueError("kind=evidence 的段落必须绑定至少一条 Evidence。")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("段落不能重复绑定同一条 Evidence。")
        return self


class MarkdownReportDraft(BaseModel):
    """Writer 的 Markdown 草稿；引用以内部 cite 标签标记。"""

    markdown: str = Field(default="", max_length=24_000)
    selected_evidence_ids: list[str] = Field(default_factory=list, max_length=24)
