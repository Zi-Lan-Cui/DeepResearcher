"""Evidence 数据契约。"""

from typing import Literal

from pydantic import BaseModel, Field


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
    retrieval_method: Literal["origin_fetch", "tavily_raw_content", "search_summary"] = (
        "origin_fetch"
    )
    locator: EvidenceLocator = Field(default_factory=EvidenceLocator)
    support: Literal["direct", "partial", "insufficient"] = "direct"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ExtractedEvidence(BaseModel):
    claim: str
    quote: str
    support: Literal["direct", "partial", "insufficient"] = "direct"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class EvidenceExtraction(BaseModel):
    evidences: list[ExtractedEvidence] = Field(default_factory=list)
