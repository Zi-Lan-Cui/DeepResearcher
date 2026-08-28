"""工具调用入参与工具结果契约。

``allow_parallel`` ClassVar 标记该工具是否允许在同一模型回合内批量派发，
由中间件层（SerialToolMiddleware）读取，不是模型可见字段。
"""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from deepsearch_agent.schemas.reporting import ReportBrief
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
