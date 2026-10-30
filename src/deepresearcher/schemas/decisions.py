"""模型结构化输出的决策 Schema：路由器与整体审阅两处。

这两个类都经 nodes 注入的 invoke_structured 取结构化输出，是生产契约。
Researcher 的决策不走这里：它以工具调用直接表达（SearchSources / ReadSources 等），
不需要一个把六种动作塞进同一个类的宽模型。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from deepresearcher.schemas.limits import STRUCTURED_COLLECTION_HARD_LIMIT
from deepresearcher.schemas.sections import ReviewIssue


class RouteDecision(BaseModel):
    # 只裁决"直答还是深研"；是否需要澄清由 Clarifier 子图自行判断，路由不设第三种交付形态。
    route: Literal["quick_answer", "deep_research"]
    reason: str


class ReviewDecision(BaseModel):
    """整体审阅只报告问题；流程根据 fatal 问题决定是否退回。"""

    # 只服务阶段卡展示；改稿依据是 feedback 与 issues，不从本字段推导。
    summary: str = ""
    feedback: str
    gaps: list[str] = Field(default_factory=list, max_length=STRUCTURED_COLLECTION_HARD_LIMIT)
    issues: list[ReviewIssue] = Field(default_factory=list)
