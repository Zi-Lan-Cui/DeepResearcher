"""来源的可解释结构化画像与轻量 URL 派生信息。"""

from typing import Literal
from urllib.parse import urlsplit

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


def source_domain(url: str) -> str:
    """从 URL 派生稳定的小写域名；供不需要完整 URL 的 Agent 视图复用。"""
    parsed = urlsplit(url if "://" in url else f"//{url}")
    return (parsed.hostname or "").lower().rstrip(".")
