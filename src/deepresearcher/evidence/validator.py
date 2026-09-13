"""Evidence 的确定性来源校验。"""

import re

from deepresearcher.evidence.models import Evidence
from deepresearcher.tools.web.parsing.models import DocumentBlock


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def validate_evidence(evidence: Evidence, blocks: list[DocumentBlock]) -> Evidence:
    """确认 quote 来自候选原文；不通过时抛出 ValueError。"""
    normalized_quote = normalize_text(evidence.quote)
    if not normalized_quote:
        raise ValueError("Evidence quote 不能为空")
    matching = [
        block for block in blocks if normalized_quote in normalize_text(block.get("text", ""))
    ]
    if not matching:
        raise ValueError("Evidence quote 不存在于候选原文")
    evidence.locator.block_ids = [block.get("block_id", "") for block in matching]
    evidence.locator.heading_path = matching[0].get("heading_path", [])
    return evidence
