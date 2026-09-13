"""State 分区、运行生命周期与节点结果契约。

``StopReason`` 是本模块的词汇单一来源：赋值端（Supervisor 及其工具）与
消费端（兜底判定、面向用户的描述文案）都引用枚举成员，杜绝两处
手工维护字符串清单的漂移。
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from deepresearcher.errors import AgentError
from deepresearcher.evidence.models import Evidence
from deepresearcher.routing import NodeName
from deepresearcher.schemas.reporting import (
    Citation,
    ParagraphBinding,
    ReportBrief,
    ResearchSynthesis,
    WriterDirective,
)


class StopReason(StrEnum):
    """Supervisor 研究停止原因的唯一词汇来源。"""

    SUFFICIENT = "supervisor_sufficient"
    SUFFICIENT_WITHOUT_EVIDENCE = "sufficient_without_evidence"
    NO_NEW_TASKS = "no_new_tasks"
    NO_TOOL_CALLS = "no_tool_calls"
    ROUND_BUDGET_EXHAUSTED = "round_budget_exhausted"
    GLOBAL_ROUND_BUDGET_EXHAUSTED = "global_round_budget_exhausted"
    MODEL_CALL_LIMIT_EXCEEDED = "supervisor_model_call_limit_exceeded"
    AGENT_FAILED = "supervisor_agent_failed"

    @property
    def allows_partial_report(self) -> bool:
        """达到材料安全线后，允许以部分报告兜底进入 Writer 的终止原因。"""
        return self in {
            StopReason.ROUND_BUDGET_EXHAUSTED,
            StopReason.GLOBAL_ROUND_BUDGET_EXHAUSTED,
            StopReason.MODEL_CALL_LIMIT_EXCEEDED,
            StopReason.NO_NEW_TASKS,
        }

    @property
    def description(self) -> str:
        return _STOP_REASON_DESCRIPTIONS.get(
            self, "Supervisor 未确认现有材料足以形成完整研究报告。"
        )


_STOP_REASON_DESCRIPTIONS: dict[StopReason, str] = {
    StopReason.SUFFICIENT_WITHOUT_EVIDENCE: "充分性决策与 Evidence 状态矛盾。",
    StopReason.NO_NEW_TASKS: "没有可去重的新研究任务。",
    StopReason.NO_TOOL_CALLS: "Supervisor 模型既未派发研究任务，也未给出充分性决策。",
    StopReason.ROUND_BUDGET_EXHAUSTED: "研究轮次预算已耗尽。",
    StopReason.GLOBAL_ROUND_BUDGET_EXHAUSTED: "研究轮次预算已耗尽，Supervisor 尚未确认材料足以成文。",
    StopReason.MODEL_CALL_LIMIT_EXCEEDED: "Supervisor 单次运行的模型调用预算已耗尽（轮内工具调用超过天花板）。",
}


class RunError(BaseModel):
    """顶层流程失败的稳定交接契约。"""

    stage: str
    code: str
    message: str
    retryable: bool = False
    detail: str = ""

    @classmethod
    def from_exception(cls, stage: str, error: Exception) -> "RunError":
        """从统一应用错误或未知异常构造稳定运行错误。"""
        if isinstance(error, AgentError):
            return cls(
                stage=stage,
                code=error.code,
                message=str(error) or error.__class__.__name__,
                retryable=error.retryable,
                detail=error.detail or error.__class__.__name__,
            )
        return cls(
            stage=stage,
            code="node_failed",
            message=str(error) or error.__class__.__name__,
            retryable=False,
            detail=error.__class__.__name__,
        )


class RunLifecycle(BaseModel):
    phase: Literal[
        "routing",
        "clarification",
        "researching",
        "writing",
        "reviewing",
        "rendering",
        "completed",
        "failed",
    ] = "routing"
    terminal_reason: str = ""
    error: RunError | None = None


class ResearchProgress(BaseModel):
    status: Literal["not_started", "running", "completed", "incomplete", "failed"] = "not_started"
    current_round: int = Field(default=0, ge=0)
    coverage_gaps: list[str] = Field(default_factory=list)
    generation_mode: Literal["not_ready", "partial", "full"] = "not_ready"
    is_sufficient: bool = False


class WriterProgress(BaseModel):
    status: Literal["not_started", "running", "completed", "failed", "exhausted"] = "not_started"
    attempts: int = Field(default=0, ge=0)
    failure_kind: str = ""
    feedback: str = ""
    selected_evidence_ids: list[str] = Field(default_factory=list)


class ReviewIssue(BaseModel):
    """审阅意见的严重级别；warning 不阻断报告交付。"""

    severity: Literal["warning", "fatal"]
    claim: str = ""
    reason: str
    suggested_revision: str = ""


class ReviewProgress(BaseModel):
    status: Literal["pending", "approved", "rejected"] = "pending"
    attempts: int = Field(default=0, ge=0)
    feedback: str = ""
    gaps: list[str] = Field(default_factory=list)
    issues: list[ReviewIssue] = Field(default_factory=list)


class ResearchDirectionResult(BaseModel):
    """一个方向级研究任务的可审计最终结果。"""

    task_id: str
    round: int = Field(ge=1)
    task_index: int = Field(default=0, ge=0)
    question: str
    research_direction: str
    # 执行生命周期与研究覆盖度分离，避免 completed 被误读为方向已解决。
    execution_status: Literal["completed", "failed", "cancelled"]
    coverage_status: Literal["sufficient", "partial", "insufficient"]
    evidence_count: int = Field(ge=0)
    source_count: int = Field(ge=0)
    conclusion: str = ""
    remaining_gaps: list[str] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)
    read_urls: list[str] = Field(default_factory=list)
    skip_reasons: list[str] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)
    stop_reason: str
    stop_detail: str = ""


class ResearchAgentResult(BaseModel):
    """ResearchAgent 完成一个方向后的完整返回契约。"""

    evidences: list[Evidence] = Field(default_factory=list)
    selected_evidence_ids: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    task_result: ResearchDirectionResult

    @model_validator(mode="after")
    def validate_selection(self) -> "ResearchAgentResult":
        archived = {item.evidence_id for item in self.evidences}
        selected = set(self.selected_evidence_ids)
        if len(selected) != len(self.selected_evidence_ids):
            raise ValueError("方向结果不能重复选择同一 Evidence。")
        if not selected.issubset(archived):
            raise ValueError("方向选择的 Evidence 必须存在于方向候选档案。")
        if self.task_result.evidence_count != len(selected):
            raise ValueError("方向结果 evidence_count 必须等于选中的 Evidence 数量。")
        return self


class SupervisorStateUpdate(BaseModel):
    """Supervisor 产出的 State 增量契约。"""

    evidences: list[Evidence] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    task_results: list[ResearchDirectionResult] = Field(default_factory=list)
    attempted_source_urls: list[str] = Field(default_factory=list)
    active_evidence_ids: list[str] = Field(default_factory=list)
    working_set_revision: int = Field(default=0, ge=0)
    research_synthesis: ResearchSynthesis | None = None
    report_brief: ReportBrief | None = None
    writer_directive: WriterDirective | None = None
    run: RunLifecycle
    research: ResearchProgress
    writer: WriterProgress
    supervisor_next: NodeName = NodeName.RENDER_FINAL_REPORT

    def state_update(self) -> dict[str, object]:
        """转换为 LangGraph 增量；生命周期状态已经在模型边界完成。"""
        return {
            "run": self.run,
            "research": self.research,
            "writer": self.writer,
            "supervisor_next": self.supervisor_next,
            "evidences": self.evidences,
            "source_refs": self.source_refs,
            "task_results": self.task_results,
            "attempted_source_urls": self.attempted_source_urls,
            "active_evidence_ids": self.active_evidence_ids,
            "working_set_revision": self.working_set_revision,
            "research_synthesis": self.research_synthesis,
            "report_brief": self.report_brief,
            "writer_directive": self.writer_directive,
        }


class WriterResult(BaseModel):
    """Writer 返回给 LangGraph State 的已校验状态增量。"""

    report: str | None = None
    citations: list[Citation] | None = None
    paragraph_bindings: list[ParagraphBinding] | None = None
    answer_mode: (
        Literal["quick_answer", "deep_research", "research_incomplete", "review_limited"] | None
    ) = None
    current_round: int | None = Field(default=None, ge=0)
    evidence_count: int | None = Field(default=None, ge=0)
    source_count: int | None = Field(default=None, ge=0)
    run: RunLifecycle | None = None
    writer: WriterProgress | None = None
    review: ReviewProgress | None = None
    writer_draft: str | None = None
    report_draft: str | None = None
    writer_selected_evidence_ids: list[str] | None = None

    def state_update(self) -> dict[str, object]:
        """转为 LangGraph 状态增量，阶段状态只通过嵌套模型交接。"""
        update = self.model_dump(exclude_none=True)
        for name in ("run", "writer", "review"):
            value = getattr(self, name)
            if value is not None:
                update[name] = value
        return update
