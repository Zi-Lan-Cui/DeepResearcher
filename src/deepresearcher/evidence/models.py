"""Evidence 数据契约与入站 ID 归一化。"""

import re

from pydantic import AliasChoices, BaseModel, Field

from deepresearcher.schemas.sources import SourceProfile
from deepresearcher.vocab import RETRIEVAL_ORIGIN_FETCH, Support

# 系统生成的 evidence_id 形如 {task_id}-ev-{hex16}，合法字符集里没有空白与控制符；
# 模型转写长 ID 时会混入 \r 之类干扰，进行归一化后再存储
_EVIDENCE_ID_NOISE_RE = re.compile(r"[\s\x00-\x1f\x7f\u200b\u2060\ufeff]+")


def normalize_evidence_id(value: str) -> str:
    """移除单条入站 evidence_id 中的空白与控制字符。"""
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


_WHITESPACE_RUN_RE = re.compile(r"\s+")


def normalize_title_whitespace(value: str) -> str:
    """把网页标题里的换行与连续空白合并为单空格：标题会被逐行消费（证据来源表），
    内嵌换行会把一行记录劈成两行，下游按行解析即整条失配。"""
    return _WHITESPACE_RUN_RE.sub(" ", value).strip()


class Evidence(BaseModel):
    evidence_id: str
    # 来源Task的 task id
    task_id: str = Field(validation_alias=AliasChoices("task_id", "task_id"))
    research_direction: str
    claim: str
    quote: str
    source_url: str
    source_title: str = ""
    # 搜索提供方给出的发布时间，用于时效性判断；不属于 quote 原文。
    published_at: str = ""
    source_profile: SourceProfile = Field(default_factory=SourceProfile)
    # 使用的检索方法
    retrieval_method: str = RETRIEVAL_ORIGIN_FETCH
    support: Support = "strict"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    def agent_payload(self) -> dict[str, object]:
        """返回可以暴露给 Agent/用户的 Evidence 视图。"""
        return self.model_dump()
