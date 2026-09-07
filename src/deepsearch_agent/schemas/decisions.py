"""模型结构化输出的决策 Schema：路由、澄清、方向探索与整体审阅。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from deepsearch_agent.schemas.sections import ReviewIssue


class RouteDecision(BaseModel):
    route: Literal["quick_answer", "deep_research", "clarify_needed"]
    reason: str


class ClarificationDecision(BaseModel):
    """澄清只补齐研究意图，不改写或缩窄用户原问题。"""

    needs_user_input: bool = False
    intent_summary: str = ""
    research_focus: list[str] = Field(default_factory=list, max_length=4)
    clarification_question: str = ""


class ResearchDirectionDecision(BaseModel):
    """ResearchAgent 的内部统一决策；来源是 ResearchDirection* 工具调用。"""

    action: Literal["search", "read", "inspect", "release", "restore", "complete"]
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
        if self.action in {"release", "restore"} and (
            not self.evidence_ids or self.queries or self.candidate_ids
        ):
            raise ValueError(
                f"action={self.action} 时必须提供 Evidence ID，且不得提供查询或候选 ID。"
            )
        if self.action == "complete" and (self.queries or self.candidate_ids or self.evidence_ids):
            raise ValueError("action=complete 时不得继续提供查询、候选 ID 或 Evidence ID。")
        if self.action != "complete" and (self.answered_points or self.conclusion.strip()):
            raise ValueError("只有 action=complete 时才能输出 answered_points 或 conclusion。")
        return self


class ReflectionDecision(BaseModel):
    """整体审阅只报告问题；流程根据 fatal 问题决定是否退回。"""

    feedback: str
    gaps: list[str] = Field(default_factory=list, max_length=6)
    issues: list[ReviewIssue] = Field(default_factory=list)
