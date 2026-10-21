"""State 分区、运行生命周期与节点结果契约。

``StopReason`` 是本模块的词汇单一来源：赋值端，即 Supervisor 及其工具，与
消费端，即兜底判定和面向用户的描述文案，都引用枚举成员，避免两处
各自维护字符串清单造成漂移。
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from deepresearcher.errors import AgentError, clip_text
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

    COMPLETED = "supervisor_completed"
    ENDED_WITHOUT_COMPLETION = "ended_without_completion"
    GLOBAL_ROUND_BUDGET_EXHAUSTED = "global_round_budget_exhausted"
    MODEL_CALL_LIMIT_EXCEEDED = "supervisor_model_call_limit_exceeded"
    PROVIDER_EXHAUSTED = "supervisor_provider_exhausted"
    AGENT_FAILED = "supervisor_agent_failed"

    @property
    def allows_partial_report(self) -> bool:
        """研究未走到显式收尾时，允许按最新工作台兜底进入 Writer 的终止原因。"""
        return self in {
            StopReason.ENDED_WITHOUT_COMPLETION,
            StopReason.GLOBAL_ROUND_BUDGET_EXHAUSTED,
            StopReason.MODEL_CALL_LIMIT_EXCEEDED,
        }

    @property
    def description(self) -> str:
        return _STOP_REASON_DESCRIPTIONS.get(
            self, "Supervisor 未确认现有材料足以形成完整研究报告。"
        )

    @property
    def rank(self) -> int:
        """终止原因的优先级数字;rank 大的覆盖 rank 小的。

        Supervisor 一次运行里多个写点都可能给 stop_reason 赋值:轮次预算、
        模型调用上限、异常、模型显式决定。此前优先级靠物理书写顺序和零散的
        is None 判断隐式规定,新增写点稍有不慎就会覆盖本该显示的终态;改为
        单调 rank 后,SupervisorLoopState.set_stop_reason 按 rank 采纳,
        新增写点只需选合适的 rank,与书写顺序无关。

        分级理由:
        - ENDED_WITHOUT_COMPLETION 是最低级:它只是"未收尾即离场"的兜底记录,
          后者是在轮次用尽之前就被强制终止。
        - GLOBAL 含义是轮次用尽,低于模型调用上限;MODEL_CALL_LIMIT 表示没有机会收尾。
        - PROVIDER_EXHAUSTED 排在调用上限与异常之间:模型确实显式收尾,
          但动因是账户级系统性不可用;同一轮若再出现真实异常,异常优先。
        - AGENT_FAILED 高于以上各项:基础设施失败需要显式暴露。
        - COMPLETED 排最高:模型显式决定收尾,优先于基础设施信号。
        """
        return _STOP_REASON_RANKS.get(self, 0)


_STOP_REASON_DESCRIPTIONS: dict[StopReason, str] = {
    # tests/test_supervisor.py 校验每个 StopReason 都登记了描述:缺描述会静默落到语义相反的兜底句。
    StopReason.COMPLETED: "Supervisor 完成研究收束并冻结综合稿；不足是否随稿披露由各板块的 disclose 决定。",
    StopReason.ENDED_WITHOUT_COMPLETION: "模型结束研究而未显式收尾，按最新工作台交付部分报告。",
    StopReason.GLOBAL_ROUND_BUDGET_EXHAUSTED: "研究轮次预算已耗尽，Supervisor 尚未确认材料足以成文。",
    StopReason.MODEL_CALL_LIMIT_EXCEEDED: "Supervisor 单次运行的模型调用预算已耗尽（轮内工具调用超过天花板）。",
    StopReason.PROVIDER_EXHAUSTED: "搜索供应商已被认证账户级不可用（额度耗尽或密钥无效），无法再获取任何新 Evidence，Supervisor 无证据收束本轮研究。",
    StopReason.AGENT_FAILED: "Supervisor 本轮执行失败，未能完成研究。",
}


# 优先级数字，见 StopReason.rank 的 docstring；间隔取 5，便于将来插入新级别。
_STOP_REASON_RANKS: dict[StopReason, int] = {
    StopReason.ENDED_WITHOUT_COMPLETION: 20,
    StopReason.GLOBAL_ROUND_BUDGET_EXHAUSTED: 25,
    StopReason.MODEL_CALL_LIMIT_EXCEEDED: 30,
    StopReason.PROVIDER_EXHAUSTED: 40,
    StopReason.AGENT_FAILED: 45,
    StopReason.COMPLETED: 50,
}


class DirectionStopReason(StrEnum):
    """方向级 ResearchAgent 的终止原因;与 StopReason 分属两域,成员值就是序列化 wire 值。"""

    COMPLETE = "complete"
    STEP_BUDGET_EXHAUSTED = "step_budget_exhausted"
    DIRECTION_AGENT_FAILED = "direction_agent_failed"
    FALLBACK_COMPLETE = "fallback_complete"
    BLOCKED_WITHOUT_EVIDENCE = "blocked_without_evidence"
    WORKER_EXCEPTION = "worker_exception"
    CANCELLED = "cancelled"


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
        # 校验器只截断不抛错,避免错误处理路径自身因超长文本再抛出新异常。
        text = str(value)
        if len(text) <= RUN_ERROR_TEXT_HARD_LIMIT_CHARS:
            return value
        return clip_text(text, RUN_ERROR_TEXT_HARD_LIMIT_CHARS)

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


def failure_event_fields(stage: str, exc: Exception) -> tuple[RunError, dict[str, Any]]:
    """失败事件与失败状态的公共字段单源。

    状态侧 node_runner 与事件侧 instrumentation 由同一个 RunError 构造,
    error/code/retryable 两处保持一致、不各自演化;link、duration 等场景
    专有字段由调用方自行补进 make_node_event。
    """
    error = RunError.from_exception(stage, exc)
    return error, {
        "error": error.message,
        "payload": {"code": error.code, "retryable": error.retryable},
    }


class RenderOutcome(StrEnum):
    """渲染终点节点写下的交付结局;terminal_reason 词汇的第二域,第一域是 StopReason。

    成员值已被 checkpoint 与前端冻结,本枚举不改变 wire,只作枚举成员与用户可读
    文案的单一来源;跨进程持久化的 lifecycle 终态,如 user_cancelled、
    server_shutdown,是第三域,仅登记于 DB 侧,不经此枚举。
    """

    QUICK_ANSWER = "quick_answer"
    WRITER_EXHAUSTED = "writer_exhausted"
    RESEARCH_INCOMPLETE = "research_incomplete"
    REVIEW_RECOVERY_EXHAUSTED = "review_recovery_exhausted"
    REVIEW_REJECTED = "review_rejected"
    MISSING_REPORT_DRAFT = "missing_report_draft"
    REPORT_RENDERED = "report_rendered"

    @property
    def description(self) -> str:
        return _RENDER_OUTCOME_DESCRIPTIONS[self]


_RENDER_OUTCOME_DESCRIPTIONS: dict[RenderOutcome, str] = {
    RenderOutcome.QUICK_ANSWER: "以即时回答方式交付。",
    RenderOutcome.WRITER_EXHAUSTED: "报告写作未通过引用协议校验。",
    RenderOutcome.RESEARCH_INCOMPLETE: "研究未完成。",
    RenderOutcome.REVIEW_RECOVERY_EXHAUSTED: "审阅修订次数已耗尽。",
    RenderOutcome.REVIEW_REJECTED: "整体审阅未通过。",
    RenderOutcome.MISSING_REPORT_DRAFT: "缺少可渲染的草稿或引用元数据。",
    RenderOutcome.REPORT_RENDERED: "报告已渲染交付。",
}


def terminal_reason_text(value: object) -> str:
    """把 terminal_reason 换成面向用户的中文,三域词汇的统一查表出口。

    terminal_reason 混装 StopReason 值、RenderOutcome 值与 DB lifecycle 字符串;
    凡渲染进用户报告的 terminal_reason 都经此函数,机器可读词不直达用户。
    未登记的值原样返回,lifecycle 词汇在 DB 侧自有中文包装点。
    """
    text = str(value or "")
    if not text:
        return "研究未完成。"
    for enum_cls in (StopReason, RenderOutcome):
        try:
            return enum_cls(text).description
        except ValueError:
            continue
    return text


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
    # 方向结果汇总的观测缺口,供中断诊断;与板块 gaps 台账分属两路。
    coverage_gaps: list[str] = Field(default_factory=list)
    generation_mode: GenerationMode = "not_ready"


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
    # 方向卡短题，供 Supervisor 证据视图回查方向归属；老检查点缺省为空，
    # 卡片渲染回退长契约截断。
    display_title: str = ""
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
    stop_reason: DirectionStopReason
    stop_detail: str = ""
    # 数据信号，不是控制流：搜索服务账户级不可用时置真，Supervisor 模型读到后自然停止派发或收尾。
    provider_exhausted: bool = False


class PendingDirection(BaseModel):
    """在途方向的登记项：派发批随回执提交进通道，到达吸收时移除。

    subtask 存 SubTask 全量载荷，崩溃恢复后凭它把该方向判为丢失并如实报告。
    """

    task_id: str
    subtask: dict[str, object]
    display_title: str = ""
    round: int = Field(ge=1)


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
        """转换为 LangGraph 增量；生命周期校验已在模型构造边界完成。"""
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
    report_title: str | None = None
    citations: list[Citation] | None = None
    paragraph_bindings: list[ParagraphBinding] | None = None
    answer_mode: WriterAnswerMode | None = None
    current_round: int | None = Field(default=None, ge=0)
    evidence_count: int | None = Field(default=None, ge=0)
    source_count: int | None = Field(default=None, ge=0)
    run: RunStatus | None = None
    writer: WriterProgress | None = None
    review: ReviewProgress | None = None
    rejected_draft: str | None = None
    report_draft: str | None = None
    writer_selected_evidence_ids: list[str] | None = None

    def state_update(self) -> dict[str, object]:
        """转为 LangGraph 状态增量，阶段状态只通过嵌套模型交接。

        None 字段不产生增量;模型字段以对象整体写入:若用 model_dump 摊平成
        dict,同 run 内消费方拿到模型对象的前提就被破坏。dict 与模型的转换
        只发生在 checkpoint 边界。
        """
        return {name: value for name, value in self if value is not None}
