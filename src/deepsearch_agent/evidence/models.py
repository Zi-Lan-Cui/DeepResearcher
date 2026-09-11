"""Evidence 数据契约。"""

from typing import Literal

from pydantic import BaseModel, Field

from deepsearch_agent.schemas.sources import SourceProfile


class EvidenceLocator(BaseModel):
    block_ids: list[str] = Field(default_factory=list)
    heading_path: list[str] = Field(default_factory=list)


class Evidence(BaseModel):
    evidence_id: str
    subtask_id: str
    research_direction: str
    claim: str
    quote: str
    source_url: str
    source_title: str = ""
    source_profile: SourceProfile = Field(default_factory=SourceProfile)
    retrieval_method: Literal["origin_fetch", "tavily_raw_content", "search_summary"] = (
        "origin_fetch"
    )
    locator: EvidenceLocator = Field(default_factory=EvidenceLocator)
    support: Literal["direct", "partial", "insufficient"] = "direct"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    # 仅用于审计/评测抽取是否正确；随 Evidence 进入 checkpoint，
    # 但不得进入 Agent 消息或前端事件。长度由抽取器在写入时约束。
    audit_chunk: str = Field(default="", repr=False)

    def agent_payload(self) -> dict[str, object]:
        """返回可以暴露给 Agent/用户的 Evidence 视图。"""
        return self.model_dump(exclude={"audit_chunk"})


class ExtractedEvidence(BaseModel):
    claim: str
    quote: str
    support: Literal["direct", "partial", "insufficient"] = "direct"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class EvidenceExtraction(BaseModel):
    evidences: list[ExtractedEvidence] = Field(default_factory=list)
