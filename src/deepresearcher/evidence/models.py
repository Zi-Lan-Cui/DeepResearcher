"""Evidence 数据契约。"""

from typing import Literal

from pydantic import BaseModel, Field

from deepresearcher.schemas.sources import SourceProfile
from deepresearcher.vocab import Support


class Evidence(BaseModel):
    evidence_id: str
    subtask_id: str
    research_direction: str
    claim: str
    quote: str
    source_url: str
    source_title: str = ""
    # 搜索提供方给出的发布时间，用于时效性判断；不属于 quote 原文。
    published_at: str = ""
    source_profile: SourceProfile = Field(default_factory=SourceProfile)
    retrieval_method: Literal[
        "origin_fetch", "aliyun_web_fetch", "tavily_raw_content", "search_summary"
    ] = "origin_fetch"
    support: Support = "direct"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    def agent_payload(self) -> dict[str, object]:
        """返回可以暴露给 Agent/用户的 Evidence 视图。"""
        return self.model_dump()
