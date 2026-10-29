"""State 分区、运行生命周期与节点结果契约。"""

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
from deepresearcher.schemas.tasks import Task
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
        """研究未走到显式收尾时，允许按最新工作台降级进入 Writer 的终止原因。"""
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
        """终止原因的优先级数字，数值大的覆盖数值小的。

        多个写点都可能给 stop_reason 赋值，SupervisorRoundState.set_stop_reason 按 rank 采纳，
        新增写点只需选定合适的 rank，与代码书写顺序无关。
        """
        return _STOP_REASON_RANKS.get(self, 0)


_STOP_REASON_DESCRIPTIONS: dict[StopReason, str] = {
    # tests/test_supervisor.py 校验每个 StopReason 都登记了描述：缺描述会静默落到语义相反的默认句。
    StopReason.COMPLETED: "Supervisor 完成研究并冻结综合稿；不足是否随稿披露由各板块的 disclose 决定。",
    StopReason.ENDED_WITHOUT_COMPLETION: "模型结束研究而未显式收尾，按最新工作台交付部分报告。",
    StopReason.GLOBAL_ROUND_BUDGET_EXHAUSTED: "研究轮次预算已耗尽，Supervisor 尚未确认材料足以成文。",
    StopReason.MODEL_CALL_LIMIT_EXCEEDED: "Supervisor 单次运行的模型调用预算已耗尽（轮内工具调用超过上限）。",
    StopReason.PROVIDER_EXHAUSTED: "搜索供应商已被认证账户级不可用（额度耗尽或密钥无效），无法再获取任何新 Evidence，Supervisor 无证据结束本轮研究。",
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


class TaskStopReason(StrEnum):
    """任务级 Researcher 的终止原因；与 StopReason 分属两域，成员值就是序列化 wire 值。"""

    COMPLETE = "complete"
    STEP_BUDGET_EXHAUSTED = "step_budget_exhausted"
    # 字面值保留历史写法：stop_reason 已进检查点与事件，改值会让存量 run 反序列化失败。
    TASK_AGENT_FAILED = "direction_agent_failed"
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
        # 校验器只截断不抛错，避免错误处理路径自身因超长文本再抛出新异常。
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

    状态侧 node_runner 与事件侧 instrumentation 由同一个 RunError 构造，
    error/code/retryable 两处保持一致、不各自演化；link、duration 等场景
    专有字段由调用方自行补进 make_node_event。
    """
    error = RunError.from_exception(stage, exc)
    return error, {
        "error": error.message,
        "payload": {"code": error.code, "retryable": error.retryable},
    }


class RenderOutcome(StrEnum):
    """渲染终点节点写下的交付结局"""

    QUICK_ANSWER = "quick_answer"
    WRITER_EXHAUSTED = "writer_exhausted"
    RESEARCH_INCOMPLETE = "research_incomplete"
    REVIEW_RECOVERY_EXHAUSTED = "review_recovery_exhausted"
    REVIEW_REJECTED = "review_rejected"
    REVIEW_FAILED = "review_failed"
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
    RenderOutcome.REVIEW_FAILED: "审阅失败，报告未经审阅校验。",
    RenderOutcome.MISSING_REPORT_DRAFT: "缺少可渲染的草稿或引用元数据。",
    RenderOutcome.REPORT_RENDERED: "报告已渲染交付。",
}


def terminal_reason_text(value: object) -> str:
    """把 terminal_reason 换成面向用户的中文，三域词汇的统一查表出口。

    terminal_reason 混装 StopReason 值、RenderOutcome 值与 DB lifecycle 字符串；
    凡渲染进用户报告的 terminal_reason 都经此函数，机器可读词不直达用户。
    未登记的值原样返回，lifecycle 词汇在 DB 侧自有中文包装点。
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
    status: Literal["pending", "approved", "rejected", "failed"] = "pending"
    attempts: int = Field(default=0, ge=0)
    feedback: str = ""
    gaps: list[str] = Field(default_factory=list)
    issues: list[ReviewIssue] = Field(default_factory=list)
    failure_reason: str = ""


class TaskResult(BaseModel):
    """一个研究任务的可审计最终结果。"""

    task_id: str
    round: int = Field(ge=1)
    task_index: int = Field(default=0, ge=0)
    question: str
    research_direction: str
    # 任务卡短题，供 Supervisor 证据视图回查任务归属；老检查点缺省为空，
    # 卡片渲染回退长契约截断。
    display_title: str = ""
    # 执行生命周期与研究覆盖度分离，避免 completed 被误读为任务已解决。
    execution_status: Literal["completed", "failed", "cancelled"]
    coverage_status: Literal["sufficient", "partial", "insufficient"]
    evidence_count: int = Field(ge=0)
    source_count: int = Field(ge=0)
    # 方向落定时对当前状态的唯一叙述：发现、已查清与未查清、停止原因都在这一段里。
    # 不拆结构化缺口清单——缺口脱离上下文后只剩短语，下游无法据此判断。
    conclusion: str = ""
    # 只服务任务卡展示：本方向查清了什么，两句话以内。Supervisor 读的是 conclusion，
    # 缺口仍留在那里，不从本字段推导；旧检查点缺省为空，卡面不渲染摘要行。
    finding: str = ""
    queries: list[str] = Field(default_factory=list)
    read_urls: list[str] = Field(default_factory=list)
    skip_reasons: list[str] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)
    stop_reason: TaskStopReason
    stop_detail: str = ""
    # 数据信号，不是控制流：搜索服务账户级不可用时置真，Supervisor 模型读到后自然停止派发或收尾。
    provider_exhausted: bool = False


class PendingTask(BaseModel):
    """在途任务的登记项：派发批随回执提交进通道，到达吸收时移除。

    payload 存 Task 全量载荷，崩溃恢复后凭它把该任务判为丢失并如实报告。
    """

    task_id: str
    payload: dict[str, object]
    display_title: str = ""
    round: int = Field(ge=1)


class ResearcherResult(BaseModel):
    """Researcher 完成一个任务后的完整返回契约。

    成功、worker 异常、崩溃恢复丢失三条路径都返回本对象，`asyncio.Task` 的 await
    方因此总能直接取值：失败不是异常，是一个 execution_status 为 failed 的正常结果。
    """

    evidences: list[Evidence] = Field(default_factory=list)
    selected_evidence_ids: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    task_result: TaskResult

    @classmethod
    def failed_for(cls, task: Task, round_no: int, error: str) -> "ResearcherResult":
        """构造 worker 异常或丢失的降级结果；失败经任务报告让 Supervisor 模型可见。"""
        return cls(
            task_result=TaskResult(
                task_id=task["id"],
                round=round_no,
                task_index=int(task.get("sequence", 0)),
                question=task["question"],
                research_direction=task["question"],
                display_title=str(task.get("display_title") or ""),
                execution_status="failed",
                coverage_status="insufficient",
                evidence_count=0,
                source_count=0,
                failures=[error],
                # 失败路径也要给 Supervisor 一段可读的状态说明：tasks 归档只带 conclusion，
                # failures 不在其中，空结论会让这条失败在上下文压缩后彻底消失。
                conclusion=f"本任务未产出研究结论：{error}。该方向仍是缺口，需要重新派发。",
                stop_reason=TaskStopReason.WORKER_EXCEPTION,
                stop_detail=error,
            ),
        )

    @model_validator(mode="after")
    def validate_selection(self) -> "ResearcherResult":
        archived = {item.evidence_id for item in self.evidences}
        selected = set(self.selected_evidence_ids)
        if len(selected) != len(self.selected_evidence_ids):
            raise ValueError("任务结果不能重复选择同一 Evidence。")
        if not selected.issubset(archived):
            raise ValueError("任务选择的 Evidence 必须存在于任务候选档案。")
        if self.task_result.evidence_count != len(selected):
            raise ValueError("任务结果 evidence_count 必须等于选中的 Evidence 数量。")
        return self


class SupervisorStateUpdate(BaseModel):
    """Supervisor 产出的 State 增量契约。"""

    evidences: list[Evidence] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    task_results: list[TaskResult] = Field(default_factory=list)
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

        None 字段不产生增量；模型字段必须整体写入，不得用 model_dump 摊平成 dict——
        同 run 内消费方依赖拿到模型对象，dict 与模型的转换只发生在 checkpoint 边界。
        """
        return {name: value for name, value in self if value is not None}
