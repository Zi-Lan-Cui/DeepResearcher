"""State 分区、运行生命周期与节点结果契约。

``StopReason`` 是本模块的词汇单一来源：赋值端（Supervisor 及其工具）与
消费端（兜底判定、面向用户的描述文案）都引用枚举成员，杜绝两处
手工维护字符串清单的漂移。
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from deepresearcher.errors import AgentError
from deepresearcher.evidence.models import Evidence
from deepresearcher.routing import NodeName
from deepresearcher.schemas.limits import RUN_ERROR_TEXT_HARD_LIMIT_CHARS
from deepresearcher.schemas.reporting import (
    Citation,
    ParagraphBinding,
    ResearchSynthesis,
    WriterDirective,
)
from deepresearcher.vocab import GenerationMode, ResearchStatus, WriterAnswerMode


class StopReason(StrEnum):
    """Supervisor 研究停止原因的唯一词汇来源。"""

    SUFFICIENT = "supervisor_sufficient"
    SUBMITTED_WITH_GAPS = "supervisor_submitted_with_gaps"
    SUFFICIENT_WITHOUT_EVIDENCE = "sufficient_without_evidence"
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
            StopReason.SUBMITTED_WITH_GAPS,
        }

    @property
    def description(self) -> str:
        return _STOP_REASON_DESCRIPTIONS.get(
            self, "Supervisor 未确认现有材料足以形成完整研究报告。"
        )

    @property
    def rank(self) -> int:
        """终止原因的**声明式**权威度;数字越大越能覆盖小的。

        Supervisor 一次运行里多个位置可能给 stop_reason 赋值(轮次预算、
        模型调用天花板、异常、模型显式决定);过去依赖"物理书写顺序 +
        零散 is None 守卫"隐式规定优先级,任何新增写点都可能悄悄把该显示的终态
        压下去。改为单调 rank 之后,`SupervisorLoopState.set_stop_reason` 按 rank 采纳,
        新增写点不用再操心顺序——只需选一个合适的 rank。

        分级理由:
        - 轮次预算 (ROUND < GLOBAL) 表示"没做完但到点了",低于天花板命中,
          因为后者是"到点前就被强行掐掉"这一更强的终止事实。
        - 模型调用天花板 (MODEL_CALL_LIMIT) 表示"根本没机会收尾"。
        - 异常 (AGENT_FAILED) 高于以上,基础设施失败要显式暴露。
        - 两个显式决定 (SUBMITTED_WITH_GAPS < SUFFICIENT) 最高——模型自己
          说了"就这样收尾",压过一切基础设施信号;SUFFICIENT 再强于带缺口。
        """
        return _STOP_REASON_RANKS.get(self, 0)


_STOP_REASON_DESCRIPTIONS: dict[StopReason, str] = {
    # 全覆盖由 test_sections 钉死:缺描述会静默落到语义相反的兜底句。
    StopReason.SUFFICIENT: "Supervisor 确认现有材料足以形成完整研究报告。",
    StopReason.SUBMITTED_WITH_GAPS: "Supervisor 已提交带明确缺口的最新研究综合稿。",
    StopReason.SUFFICIENT_WITHOUT_EVIDENCE: "充分性决策与 Evidence 状态矛盾。",
    StopReason.ROUND_BUDGET_EXHAUSTED: "研究轮次预算已耗尽。",
    StopReason.GLOBAL_ROUND_BUDGET_EXHAUSTED: "研究轮次预算已耗尽，Supervisor 尚未确认材料足以成文。",
    StopReason.MODEL_CALL_LIMIT_EXCEEDED: "Supervisor 单次运行的模型调用预算已耗尽（轮内工具调用超过天花板）。",
    StopReason.AGENT_FAILED: "Supervisor 本轮执行失败，未能完成研究。",
}


# 声明式权威度；见 StopReason.rank 的 docstring。留 5 的间隔便于将来插入新级别。
_STOP_REASON_RANKS: dict[StopReason, int] = {
    StopReason.ROUND_BUDGET_EXHAUSTED: 20,
    StopReason.GLOBAL_ROUND_BUDGET_EXHAUSTED: 25,
    StopReason.MODEL_CALL_LIMIT_EXCEEDED: 30,
    StopReason.SUFFICIENT_WITHOUT_EVIDENCE: 40,
    StopReason.AGENT_FAILED: 45,
    StopReason.SUBMITTED_WITH_GAPS: 50,
    StopReason.SUFFICIENT: 60,
}


class RunError(BaseModel):
    """顶层流程失败的稳定交接契约。"""

    stage: str
    code: str
    message: str
    retryable: bool = False
    detail: str = ""

    @field_validator("message", "detail", mode="before")
    @classmethod
    def _clip_long_text(cls, value: object) -> object:
        # 校验器只截断不抛错:错误处理路径自身绝不允许因超长文本再造异常。
        text = str(value)
        if len(text) > RUN_ERROR_TEXT_HARD_LIMIT_CHARS:
            return text[: RUN_ERROR_TEXT_HARD_LIMIT_CHARS - 8] + "…[截断]"
        return value

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


class RunStatus(BaseModel):
    """一次 run 的状态快照：当前阶段 + 终止原因 + 错误。值对象，不驱动流程、
    不代表"正在运行的生命周期"——阶段推进由节点产出增量、Graph 路由决定。"""

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


class SupervisorProgress(BaseModel):
    status: ResearchStatus = "not_started"
    current_round: int = Field(default=0, ge=0)
    coverage_gaps: list[str] = Field(default_factory=list)
    generation_mode: GenerationMode = "not_ready"
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
    # 数据信号（非控制流）：搜索服务账户级不可用时置真，供 Supervisor 模型读到后自然停止派发/收尾。
    provider_exhausted: bool = False


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
    active_evidence_ids: list[str] = Field(default_factory=list)
    working_set_revision: int = Field(default=0, ge=0)
    research_synthesis: ResearchSynthesis | None = None
    writer_directive: WriterDirective | None = None
    run: RunStatus
    supervisor: SupervisorProgress
    writer: WriterProgress
    supervisor_next: NodeName = NodeName.RENDER_FINAL_REPORT

    def state_update(self) -> dict[str, object]:
        """转换为 LangGraph 增量；生命周期状态已经在模型边界完成。"""
        return {
            "run": self.run,
            "supervisor": self.supervisor,
            "writer": self.writer,
            "supervisor_next": self.supervisor_next,
            "evidences": self.evidences,
            "source_refs": self.source_refs,
            "task_results": self.task_results,
            "active_evidence_ids": self.active_evidence_ids,
            "working_set_revision": self.working_set_revision,
            "research_synthesis": self.research_synthesis,
            "writer_directive": self.writer_directive,
        }


class WriterResult(BaseModel):
    """Writer 返回给 LangGraph State 的已校验状态增量。"""

    report: str | None = None
    citations: list[Citation] | None = None
    paragraph_bindings: list[ParagraphBinding] | None = None
    answer_mode: WriterAnswerMode | None = None
    current_round: int | None = Field(default=None, ge=0)
    evidence_count: int | None = Field(default=None, ge=0)
    source_count: int | None = Field(default=None, ge=0)
    run: RunStatus | None = None
    writer: WriterProgress | None = None
    review: ReviewProgress | None = None
    writer_draft: str | None = None
    report_draft: str | None = None
    writer_selected_evidence_ids: list[str] | None = None

    def state_update(self) -> dict[str, object]:
        """转为 LangGraph 状态增量，阶段状态只通过嵌套模型交接。

        None 字段不产生增量;模型字段以对象整体写入——model_dump 会把
        citations/bindings 摊平成 dict,同 run 内破坏"消费方拿到的恒是模型"
        的恢复前提,dict↔模型转换只发生在 checkpoint 边界。
        """
        return {name: value for name, value in self if value is not None}
