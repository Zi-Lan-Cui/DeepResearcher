"""报告域契约：写作任务书、引用元数据与段落绑定。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from deepresearcher.evidence.models import normalize_evidence_ids
from deepresearcher.schemas.limits import (
    EVIDENCE_REFERENCES_HARD_LIMIT,
    REPORT_CAVEATS_HARD_LIMIT,
    REPORT_MARKDOWN_HARD_LIMIT_CHARS,
    STRUCTURED_COLLECTION_HARD_LIMIT,
    STRUCTURED_IDENTIFIER_HARD_LIMIT_CHARS,
    STRUCTURED_SUMMARY_HARD_LIMIT_CHARS,
    STRUCTURED_TEXT_HARD_LIMIT_CHARS,
)
from deepresearcher.schemas.sources import SourceProfile
from deepresearcher.vocab import GenerationMode, ResearchStatus, Support


class AspectIssue(BaseModel):
    """板块登记表上的一条缺口或判定标准冲突；disclose 决定它是否进任务书告知读者。"""

    text: str = Field(min_length=1, max_length=STRUCTURED_TEXT_HARD_LIMIT_CHARS)
    disclose: bool = Field(
        default=True,
        description=(
            "是否随任务书展示给读者。为 true 的条目原文进入报告 caveats 字段，是公开声明；"
            "缺口影响结论成立，或矛盾不可裁定且触及核心结论时为 true；"
            "结论已有可信次级支撑、只缺一手锚定时为 false，属可优化项。"
        ),
    )


class ReportChapter(BaseModel):
    """任务书里由 aspect 投影出的章节卡：本章结论、要点清单与各自的支撑证据。"""

    topic: str
    role: str
    required: bool = True
    conclusion: str = ""
    key_points: list[str] = Field(default_factory=list, max_length=STRUCTURED_COLLECTION_HARD_LIMIT)
    evidence_ids: list[str] = Field(default_factory=list, max_length=EVIDENCE_REFERENCES_HARD_LIMIT)


class ReportBrief(BaseModel):
    """Supervisor 交给 Writer 的任务书；不携带 Evidence 正文。"""

    answer_goal: str
    # 全文核心主张：Writer 组织报告的主线；系统回退交接时可为空串，Writer 自行归纳。
    thesis: str = ""
    chapters: list[ReportChapter] = Field(min_length=1, max_length=STRUCTURED_COLLECTION_HARD_LIMIT)
    # 投影过滤结果：各板块 gaps/conflicts 中 disclose=true 的条目。
    # 日常裁剪由 AGENT_REPORT_MAX_CAVEATS 控制；这里仅防御异常膨胀。
    caveats: list[str] = Field(default_factory=list, max_length=REPORT_CAVEATS_HARD_LIMIT)


class PlanItem(BaseModel):
    """工作台计划清单的一条；顺序即列表位置，不另设 order 字段。"""

    content: str = Field(min_length=1, max_length=STRUCTURED_TEXT_HARD_LIMIT_CHARS)
    status: Literal["not_started", "in_progress", "done"] = "not_started"


class ResearchAspect(BaseModel):
    """Supervisor 跨研究方向建立的证据支撑认知单元。

    登记表面用于覆盖管理与收尾校验；交接时逐块投影为任务书章节卡，
    aspect_id 与 status 留在登记表面不带入写作契约。
    """

    aspect_id: str = Field(min_length=1, max_length=STRUCTURED_IDENTIFIER_HARD_LIMIT_CHARS)
    topic: str = Field(min_length=1, max_length=STRUCTURED_TEXT_HARD_LIMIT_CHARS)
    role: str = Field(min_length=1, max_length=STRUCTURED_TEXT_HARD_LIMIT_CHARS)
    required: bool = True
    status: Literal["covered", "partial", "uncovered", "conflicted"]
    conclusion: str = Field(default="", max_length=STRUCTURED_SUMMARY_HARD_LIMIT_CHARS)
    key_points: list[str] = Field(default_factory=list, max_length=STRUCTURED_COLLECTION_HARD_LIMIT)
    evidence_ids: list[str] = Field(default_factory=list, max_length=EVIDENCE_REFERENCES_HARD_LIMIT)
    gaps: list[AspectIssue] = Field(
        default_factory=list,
        max_length=STRUCTURED_COLLECTION_HARD_LIMIT,
        description="本板块尚未解决的缺口；动态更新——补齐或转为可优化后重提本 aspect 改写记录，不留过期条目。",
    )
    conflicts: list[AspectIssue] = Field(
        default_factory=list,
        max_length=STRUCTURED_COLLECTION_HARD_LIMIT,
        description="本板块证据间的结论冲突；裁定或有新证据后同样重提改写。",
    )

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_fields(cls, data: object) -> object:
        """旧 checkpoint 结构兼容：summary 由 conclusion 承接，remaining_gap 并入首条 gap。"""
        if isinstance(data, dict) and ("summary" in data or "remaining_gap" in data):
            data = dict(data)
            if "conclusion" not in data and "summary" in data:
                data["conclusion"] = data.pop("summary")
            else:
                data.pop("summary", None)
            legacy_gap = str(data.pop("remaining_gap", "") or "").strip()
            if legacy_gap:
                # 综合稿级的并入先于本校验运行，可能已占用 gaps；旧缺口前置合并保序。
                data["gaps"] = [
                    {"text": legacy_gap, "disclose": True},
                    *(data.get("gaps") or []),
                ]
        return data

    @field_validator("evidence_ids", mode="before")
    @classmethod
    def normalize_evidence_id_lists(cls, value: object) -> object:
        return normalize_evidence_ids(value)

    @model_validator(mode="after")
    def validate_grounding(self) -> "ResearchAspect":
        if self.status == "covered" and not self.evidence_ids:
            raise ValueError("covered 研究方面必须绑定至少一条 Evidence。")
        if self.status == "uncovered" and self.evidence_ids:
            raise ValueError("uncovered 研究方面不能绑定 Evidence。")
        if self.status in {"partial", "uncovered", "conflicted"} and not self.gaps:
            raise ValueError(f"{self.status} 研究方面必须登记至少一条 gap。")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("同一研究方面不能重复绑定 Evidence。")
        return self


class ResearchSynthesis(BaseModel):
    """Supervisor 的结构化研究综合状态。

    记录当前工作集下各研究方面的覆盖、证据绑定、缺口与冲突。它不是最终报告，
    也不是某个 ResearchAgent 的方向结果；只有经 ResearchComplete 校验并冻结后，
    才成为交给 Writer 的交接版本。
    """

    revision: int = Field(ge=1)
    based_on_working_set_revision: int = Field(ge=0)
    # 草稿期允许先立一块 aspect 再逐步补齐；answer_goal/thesis/decision_rationale 齐备
    # 是 ResearchComplete 的校验项(freeze_synthesis)，不是入场项。
    answer_goal: str = Field(default="", max_length=STRUCTURED_TEXT_HARD_LIMIT_CHARS)
    thesis: str = Field(default="", max_length=STRUCTURED_TEXT_HARD_LIMIT_CHARS)
    aspects: list[ResearchAspect] = Field(max_length=STRUCTURED_COLLECTION_HARD_LIMIT)
    selected_evidence_ids: list[str] = Field(
        default_factory=list,
        max_length=EVIDENCE_REFERENCES_HARD_LIMIT,
        description="由 aspects.evidence_ids 按首次出现顺序派生，不由模型独立决定。",
    )
    decision_rationale: str = Field(default="", max_length=STRUCTURED_TEXT_HARD_LIMIT_CHARS)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_global_issues(cls, data: object) -> object:
        """旧 checkpoint 结构兼容：全局 open_gaps/conflicts 并入首块 aspect 的 gaps 登记表。"""
        if isinstance(data, dict) and ("open_gaps" in data or "conflicts" in data):
            data = dict(data)
            orphans = [
                {"text": str(text).strip(), "disclose": True}
                for text in [
                    *(data.pop("open_gaps", None) or []),
                    *(data.pop("conflicts", None) or []),
                ]
                if str(text).strip()
            ]
            aspects = data.get("aspects") or []
            if orphans and aspects and isinstance(aspects[0], dict):
                aspects[0] = {**aspects[0], "gaps": [*(aspects[0].get("gaps") or []), *orphans]}
                data["aspects"] = aspects
        return data

    @model_validator(mode="after")
    def validate_selection(self) -> "ResearchSynthesis":
        aspect_ids = [item.aspect_id for item in self.aspects]
        if len(set(aspect_ids)) != len(aspect_ids):
            raise ValueError("研究综合稿不能包含重复 aspect_id。")
        selected = list(
            dict.fromkeys(
                evidence_id for aspect in self.aspects for evidence_id in aspect.evidence_ids
            )
        )
        if len(selected) > EVIDENCE_REFERENCES_HARD_LIMIT:
            raise ValueError(
                f"研究综合稿异常膨胀：Evidence 引用超过 {EVIDENCE_REFERENCES_HARD_LIMIT} 条。"
            )
        self.selected_evidence_ids = selected
        return self


class WriterDirective(BaseModel):
    """Supervisor 交给 Writer 的唯一写作交接契约。"""

    query: str
    report_brief: ReportBrief
    research_status: ResearchStatus
    generation_mode: GenerationMode
    evidence_ids: list[str] | None = None
    # 日常裁剪与 ReportBrief caveats 共用配置；这里仅防御异常膨胀。
    known_gaps: list[str] = Field(default_factory=list, max_length=REPORT_CAVEATS_HARD_LIMIT)
    revision_instructions: list[str] = Field(
        default_factory=list, max_length=STRUCTURED_COLLECTION_HARD_LIMIT
    )
    previous_draft: str = ""


class Citation(BaseModel):
    """渲染后正文引用所需的可审计来源元数据。"""

    id: str
    url: str = ""
    title: str = ""
    quote: str = ""
    claim: str = ""
    support: Support = "direct"
    published_at: str = ""
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
    """Writer Markdown 草稿的容量上限；测试替身按本结构造草稿，生产提交走 writer 工具的 CompleteReport。"""

    markdown: str = Field(default="", max_length=REPORT_MARKDOWN_HARD_LIMIT_CHARS)
    selected_evidence_ids: list[str] = Field(
        default_factory=list, max_length=EVIDENCE_REFERENCES_HARD_LIMIT
    )
