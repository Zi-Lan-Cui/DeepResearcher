"""Evidence 数据契约与入站 ID 归一化。"""

import re

from pydantic import BaseModel, Field

from deepresearcher.schemas.sources import SourceProfile
from deepresearcher.vocab import RETRIEVAL_ORIGIN_FETCH, Support

# 系统生成的 evidence_id 形如 {task_id}-ev-{hex16}，合法字符集里没有空白与控制符；
# 模型转写长 ID 时会混入 \r 之类噪声，入站剥除这些字符是无损归一化。
_EVIDENCE_ID_NOISE_RE = re.compile(r"[\s\x00-\x1f\x7f\u200b\u2060\ufeff]+")


def normalize_evidence_id(value: str) -> str:
    """剥除单个入站 evidence_id 中的空白与控制字符。"""
    return _EVIDENCE_ID_NOISE_RE.sub("", value)


def normalize_evidence_ids(values: object) -> object:
    """入站 ID 列表统一处理：逐条归一化并按原顺序去重；非列表原样交回给字段校验报错。"""
    if not isinstance(values, list):
        return values
    normalized: list[object] = []
    for item in values:
        cleaned = normalize_evidence_id(item) if isinstance(item, str) else item
        if cleaned not in normalized:
            normalized.append(cleaned)
    return normalized


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
    # 来源方法是自由字符串，当前词表在 vocab.RETRIEVAL_*；新 provider 经 fetch
    # 编排层拼接产生的新值必须原样保留，词表外的值不应导致 Evidence 构造报错。
    retrieval_method: str = RETRIEVAL_ORIGIN_FETCH
    support: Support = "direct"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    def agent_payload(self) -> dict[str, object]:
        """返回可以暴露给 Agent/用户的 Evidence 视图。"""
        return self.model_dump()
