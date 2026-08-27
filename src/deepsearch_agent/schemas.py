from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, Field, model_validator

from deepsearch_agent.errors import AgentError
from deepsearch_agent.evidence.models import Evidence


class RouteDecision(BaseModel):
    route: Literal["quick_answer", "deep_research", "clarify_needed"]
    reason: str


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


class ClarificationDecision(BaseModel):
    """澄清只补齐研究意图，不改写或缩窄用户原问题。"""

    needs_user_input: bool = False
    intent_summary: str = ""
    research_focus: list[str] = Field(default_factory=list, max_length=4)
    clarification_question: str = ""


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


class SearchSources(BaseModel):
    """ResearchAgent 请求发现当前方向的候选来源。"""

    allow_parallel: ClassVar[bool] = False

    reason: str = Field(description="为什么这些检索式能补足当前方向的证据。")
    queries: list[str] = Field(
        min_length=1,
        max_length=2,
        description="一到两条针对当前方向缺口的短检索式。",
    )


class ReadSources(BaseModel):
    """ResearchAgent 从候选目录中选择实际读取的来源。"""

    allow_parallel: ClassVar[bool] = False

    candidate_ids: list[str] = Field(
        min_length=1,
        max_length=8,
        description="要读取的候选来源 ID；只能使用 SearchSources 返回的 ID。",
    )
    reason: str = Field(description="说明这些来源与当前方向缺口的关系。")


class ReadWorkingSet(BaseModel):
    """查看当前 Agent 工作集的轻量摘要。"""

    allow_parallel: ClassVar[bool] = False

    reason: str = Field(default="", description="说明需要重新检查工作集的原因。")


class ForgetEvidence(BaseModel):
    """从当前 Agent 工作集释放 Evidence；不删除全局 Evidence 档案。"""

    allow_parallel: ClassVar[bool] = False

    evidence_ids: list[str] = Field(min_length=1, max_length=8)
    reason: str = Field(description="说明这些 Evidence 为什么应从当前工作集中释放。")


class ResearchDirectionComplete(BaseModel):
    """ResearchAgent 宣布局部探索结束；不代表整项研究完成。"""

    allow_parallel: ClassVar[bool] = False

    reason: str = Field(description="为什么当前方向可以停止继续探索。")
    answered_points: list[str] = Field(default_factory=list, max_length=4)
    conclusion: str = ""
    remaining_gaps: list[str] = Field(
        default_factory=list,
        max_length=4,
        description="仅作为 Supervisor 的局部线索，不是全局缺口结论。",
    )


class ResearchDirectionDecision(BaseModel):
    """ResearchAgent 的内部统一决策；来源是 ResearchDirection* 工具调用。"""

    action: Literal["search", "read", "inspect", "forget", "complete"]
    reason: str
    queries: list[str] = Field(default_factory=list, max_length=2)
    candidate_ids: list[str] = Field(default_factory=list, max_length=8)
    evidence_ids: list[str] = Field(default_factory=list, max_length=8)
    answered_points: list[str] = Field(default_factory=list, max_length=4)
    conclusion: str = ""
    remaining_gaps: list[str] = Field(default_factory=list, max_length=4)

    @model_validator(mode="after")
    def validate_action(self) -> "ResearchDirectionDecision":
        if self.action == "search" and (not self.queries or self.candidate_ids):
            raise ValueError("action=search 时必须提供至少一条查询。")
        if self.action == "read" and (not self.candidate_ids or self.queries):
            raise ValueError("action=read 时必须提供候选 ID，且不得继续提供查询。")
        if self.action == "inspect" and (self.queries or self.candidate_ids or self.evidence_ids):
            raise ValueError("action=inspect 时不得提供查询、候选 ID 或 Evidence ID。")
        if self.action == "forget" and (
            not self.evidence_ids or self.queries or self.candidate_ids
        ):
            raise ValueError("action=forget 时必须提供 Evidence ID，且不得提供查询或候选 ID。")
        if self.action == "complete" and (
            self.queries or self.candidate_ids or self.evidence_ids
        ):
            raise ValueError("action=complete 时不得继续提供查询、候选 ID 或 Evidence ID。")
        if self.action != "complete" and (self.answered_points or self.conclusion.strip()):
            raise ValueError("只有 action=complete 时才能输出 answered_points 或 conclusion。")
        return self


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
    answered_points: list[str] = Field(default_factory=list)
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
    source_refs: list[str] = Field(default_factory=list)
    task_result: ResearchDirectionResult


class SupervisorStateUpdate(BaseModel):
    """Supervisor 产出的 State 增量契约。"""

    evidences: list[Evidence] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    task_results: list[ResearchDirectionResult] = Field(default_factory=list)
    attempted_source_urls: list[str] = Field(default_factory=list)
    active_evidence_ids: list[str] = Field(default_factory=list)
    report_brief: ReportBrief | None = None
    writer_directive: WriterDirective | None = None
    run: RunLifecycle
    research: ResearchProgress
    writer: WriterProgress
    supervisor_next: Literal["writer", "render_final_report"] = "render_final_report"

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
            "report_brief": self.report_brief,
            "writer_directive": self.writer_directive,
        }


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


class ReviewIssue(BaseModel):
    """审阅意见的严重级别；warning 不阻断报告交付。"""

    severity: Literal["warning", "fatal"]
    claim: str = ""
    reason: str
    suggested_revision: str = ""


class RunLifecycle(BaseModel):
    phase: Literal[
        "routing", "clarification", "researching", "writing", "reviewing",
        "rendering", "completed", "failed",
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


class ReviewProgress(BaseModel):
    status: Literal["pending", "approved", "rejected"] = "pending"
    attempts: int = Field(default=0, ge=0)
    feedback: str = ""
    gaps: list[str] = Field(default_factory=list)
    issues: list[ReviewIssue] = Field(default_factory=list)


class ReflectionDecision(BaseModel):
    """整体审阅只报告问题；流程根据 fatal 问题决定是否退回。"""

    feedback: str
    gaps: list[str] = Field(default_factory=list, max_length=6)
    issues: list[ReviewIssue] = Field(default_factory=list)


class WriterResult(BaseModel):
    """Writer 返回给 LangGraph State 的已校验状态增量。"""

    report: str | None = None
    citations: list[Citation] | None = None
    paragraph_bindings: list[ParagraphBinding] | None = None
    answer_mode: Literal["quick_answer", "deep_research", "research_incomplete"] | None = None
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


class ResearchDelegate(BaseModel):
    """派发方向级研究任务的工具调用 Schema；task_id 由本地程序分配，不信任模型。"""

    allow_parallel: ClassVar[bool] = True

    research_topic: str = Field(
        description=(
            "要研究的具体方向。必须包含研究对象、范围、待回答的局部问题、"
            "与已有方向的区别和完成标准；补缺时必须缩小到明确缺口，不能重述原问题。"
        )
    )


class ResearchToolResult(BaseModel):
    """ResearchAgent 完成方向后的结果，作为 ToolMessage 注入 Supervisor 上下文。"""

    question: str
    execution_status: Literal["completed", "failed", "cancelled"]
    coverage_status: Literal["sufficient", "partial", "insufficient"]
    round: int = Field(ge=1)
    task_index: int = Field(default=0, ge=0)
    evidence_count: int = Field(ge=0)
    source_count: int = Field(ge=0)
    answered_points: list[str] = Field(default_factory=list)
    conclusion: str = ""
    remaining_gaps: list[str] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)
    read_urls: list[str] = Field(default_factory=list)
    skip_reasons: list[str] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)
    stop_reason: str
    stop_detail: str = ""

    @classmethod
    def from_direction_result(cls, result: ResearchDirectionResult) -> "ResearchToolResult":
        return cls(**result.model_dump())


class ResearchComplete(BaseModel):
    """Supervisor 的终止信号:现有 Evidence 已足以成文。

    尚未充分时不调用本工具,继续用 ResearchDelegate 派发互补方向;
    是否存在"不足"这一中间态不由模型声明,而由它是否继续派发来表达。
    """

    allow_parallel: ClassVar[bool] = False

    reason: str
    report_brief: ReportBrief


class ResearchReady(BaseModel):
    """Supervisor 判断已有材料可以先形成一份带缺口声明的部分报告。"""

    allow_parallel: ClassVar[bool] = False

    reason: str = Field(description="说明为什么材料足以形成基本但可能不完整的报告。")
    report_brief: ReportBrief
