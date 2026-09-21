"""Evidence 数据契约。"""

from pydantic import BaseModel, Field

from deepresearcher.schemas.sources import SourceProfile
from deepresearcher.vocab import RETRIEVAL_ORIGIN_FETCH, Support


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
    # 来源方法是自由字符串,当前词表在 vocab.RETRIEVAL_*;新 provider 经
    # fetch 编排层拼接产生的新值必须在证据层原样存活(词表外的值不该炸构造)。
    retrieval_method: str = RETRIEVAL_ORIGIN_FETCH
    support: Support = "direct"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    def agent_payload(self) -> dict[str, object]:
        """返回可以暴露给 Agent/用户的 Evidence 视图。"""
        return self.model_dump()
