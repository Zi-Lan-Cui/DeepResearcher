"""来源的可解释结构化画像。"""

from typing import Literal

from pydantic import BaseModel


class SourceProfile(BaseModel):
    """仅表达可验证的来源属性，不用虚假精确的单一分数代替判断。"""

    source_type: Literal[
        "academic",
        "government",
        "standard",
        "organization",
        "company",
        "media",
        "blog",
        "general",
    ] = "general"
    authority_tier: Literal["primary", "secondary", "unknown"] = "unknown"
    publication_status: Literal["official", "preprint", "published", "unknown"] = "unknown"
    primary_source: bool = False
