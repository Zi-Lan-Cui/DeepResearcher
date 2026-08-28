"""报告域契约：写作任务书、引用元数据与段落绑定。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class CoveredTopic(BaseModel):
    """报告中必须处理的研究主题及其论证角色。"""

    topic: str
    role: str
    reason: str
    required: bool = True


class ReportBrief(BaseModel):
    """Supervisor 交给 Writer 的任务书；不携带 Evidence 正文。"""

    answer_goal: str
    covered_topics: list[CoveredTopic] = Field(min_length=1, max_length=6)
    required_points: list[str] = Field(default_factory=list, max_length=8)
    caveats: list[str] = Field(default_factory=list, max_length=6)


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
