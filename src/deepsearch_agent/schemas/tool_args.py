"""工具调用入参与工具结果契约。

``allow_parallel`` ClassVar 标记该工具是否允许在同一模型回合内批量派发，
由中间件层（SerialToolMiddleware）读取，不是模型可见字段。
"""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, Field, model_validator

from deepsearch_agent.schemas.reporting import ResearchAspect
from deepsearch_agent.schemas.sections import ResearchDirectionResult


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


class ReleaseEvidence(BaseModel):
    """从当前 Agent 活跃工作集释放 Evidence；不删除 Evidence 档案。"""

    allow_parallel: ClassVar[bool] = False

    evidence_ids: list[str] = Field(min_length=1, max_length=8)
    reason: str = Field(description="说明这些 Evidence 为什么应从当前工作集中释放。")


class RestoreEvidence(BaseModel):
    """将 Evidence 档案中的候选重新放回当前活跃工作集。"""

    allow_parallel: ClassVar[bool] = False

    evidence_ids: list[str] = Field(min_length=1, max_length=8)
    reason: str = Field(description="说明为什么需要重新启用这些 Evidence。")


class ResearchDirectionComplete(BaseModel):
    """ResearchAgent 提交方向结果并结束当前工具循环。"""

    allow_parallel: ClassVar[bool] = False

    reason: str = Field(description="为什么当前方向可以停止探索。")
    selected_evidence_ids: list[str] = Field(
        default_factory=list,
        max_length=12,
        description="最终推荐给 Supervisor 的当前活跃 Evidence ID。",
    )
    conclusion: str = Field(
        default="",
        description="基于最终选中 Evidence 的方向级简短综合；不得引入未取证事实。",
    )
    remaining_gaps: list[str] = Field(
        default_factory=list,
        max_length=4,
        description="局部未解问题；不代表整项研究的全局缺口。",
    )


class ResearchDelegate(BaseModel):
    """派发方向级研究任务的工具调用 Schema；task_id 由本地程序分配，不信任模型。"""

    allow_parallel: ClassVar[bool] = True

    research_topic: str = Field(
        description=(
            "要研究的具体方向。必须包含研究对象、范围、待回答的局部问题、"
            "与已有方向的区别和完成标准；补缺时必须缩小到明确缺口，不能重述原问题。"
        )
    )


class ResearchComplete(BaseModel):
    """冻结最新研究综合稿并终止研究阶段。"""

    allow_parallel: ClassVar[bool] = False

    synthesis_revision: int = Field(ge=1)
    reason: str = Field(description="为什么该综合版本已经足以形成完整报告。")


class ReviseResearchSynthesis(BaseModel):
    """提交 Supervisor 对当前研究状态的下一版完整规范化综合稿。"""

    allow_parallel: ClassVar[bool] = False

    expected_revision: int = Field(
        ge=0,
        description="当前综合稿版本；尚未建立综合稿时传 0。",
    )
    expected_working_set_revision: int = Field(
        ge=0,
        description="当前工具观察到的 Evidence/任务工作集版本。",
    )
    answer_goal: str = Field(min_length=1, max_length=2_000)
    overall_summary: str = Field(min_length=1, max_length=4_000)
    aspects: list[ResearchAspect] = Field(
        min_length=1,
        max_length=6,
        description=(
            "完整研究方面列表。covered 必须绑定 Evidence；partial、uncovered、"
            "conflicted 必须说明 remaining_gap。"
        ),
    )
    selected_evidence_ids: list[str] = Field(
        max_length=50,
        description=(
            "本版采用的 Evidence 完整并集，必须包含 aspects[*].evidence_ids 中每个 ID；"
            "readiness 为 partial_ready 或 complete_candidate 时不得为空。"
        ),
    )
    open_gaps: list[str] = Field(default_factory=list, max_length=12)
    conflicts: list[str] = Field(default_factory=list, max_length=8)
    next_actions: list[str] = Field(default_factory=list, max_length=8)
    readiness: Literal["not_ready", "partial_ready", "complete_candidate"]
    decision_rationale: str = Field(min_length=1, max_length=2_000)

    @model_validator(mode="after")
    def validate_evidence_contract(self) -> "ReviseResearchSynthesis":
        if len(set(self.selected_evidence_ids)) != len(self.selected_evidence_ids):
            raise ValueError("selected_evidence_ids 不能包含重复 Evidence ID。")
        selected = set(self.selected_evidence_ids)
        referenced = {item for aspect in self.aspects for item in aspect.evidence_ids}
        missing = sorted(referenced - selected)
        if missing:
            raise ValueError(
                "selected_evidence_ids 必须包含 aspects 引用的全部 Evidence；"
                f"当前缺少：{', '.join(missing)}。"
            )
        if self.readiness != "not_ready" and not selected:
            raise ValueError("partial_ready/complete_candidate 必须选择至少一条 Evidence。")
        return self


class ResearchToolResult(BaseModel):
    """ResearchAgent 完成方向后的结果，作为 ToolMessage 注入 Supervisor 上下文。"""

    question: str
    execution_status: Literal["completed", "failed", "cancelled"]
    coverage_status: Literal["sufficient", "partial", "insufficient"]
    round: int = Field(ge=1)
    task_index: int = Field(default=0, ge=0)
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

    @classmethod
    def from_direction_result(cls, result: ResearchDirectionResult) -> "ResearchToolResult":
        return cls(**result.model_dump())
