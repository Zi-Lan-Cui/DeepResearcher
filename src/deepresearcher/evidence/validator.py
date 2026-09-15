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


def locate_quote_lines(text: str, quote: str) -> tuple[int, int]:
    """在规范化全文中定位逐字引用覆盖的 1-based 行号。"""
    normalized_quote = normalize_text(quote)
    if not normalized_quote:
        raise ValueError("Evidence quote 不能为空")
    normalized_document, line_map = _normalize_with_line_map(text)
    start_offset = normalized_document.find(normalized_quote)
    if start_offset < 0:
        raise ValueError("Evidence quote 不存在于候选原文")
    end_offset = start_offset + len(normalized_quote) - 1
    return line_map[start_offset], line_map[end_offset]


def _normalize_with_line_map(text: str) -> tuple[str, list[int]]:
    """按 ``normalize_text`` 的语义归一化，并保留每个字符的原始行号。"""
    chars: list[str] = []
    line_map: list[int] = []
    line_number = 1
    pending_space = False
    for char in text:
        if char.isspace():
            pending_space = bool(chars)
            if char == "\n":
                line_number += 1
            continue
        if pending_space:
            chars.append(" ")
            line_map.append(line_number)
            pending_space = False
        lowered = char.lower()
        chars.extend(lowered)
        line_map.extend([line_number] * len(lowered))
    return "".join(chars), line_map
