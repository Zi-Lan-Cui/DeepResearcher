"""Evidence 的确定性来源校验：按首尾片段锚在原文中定位，程序逐字回填 quote。

锚点与原文归一为词序列（忽略空白与标点样式）后匹配，返回的 quote 是原文严格子串。
失败以理由码回执：empty_anchor / anchor_not_found / ambiguous_start /
end_not_found / span_too_long。
"""

import difflib
import re
import unicodedata
from dataclasses import dataclass


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _is_word_char(char: str) -> bool:
    return char.isalnum() or "一" <= char <= "鿿"


def _wordonly_scan(text: str) -> tuple[str, list[tuple[int, int, int, int]]]:
    """只保留字母/数字/CJK 做归一（NFKC + casefold，剔标点/引号/破折号/省略号/零宽），
    并行返回逐字符的键区间映射 [(键起， 键止， 源起， 源止)]，供命中后回取原文子串。"""
    pieces: list[str] = []
    spans: list[tuple[int, int, int, int]] = []
    cursor = 0
    for position, char in enumerate(text):
        fragment = unicodedata.normalize("NFKC", char).casefold()
        kept = "".join(ch for ch in fragment if _is_word_char(ch))
        if not kept:
            continue
        pieces.append(kept)
        spans.append((cursor, cursor + len(kept), position, position + 1))
        cursor += len(kept)
    return "".join(pieces), spans


def nearest_source_passage(source_text: str, quote: str, max_chars: int = 200) -> str | None:
    """返回与改述引用词面重叠最高的原文句子，供模型逐字替换为正确的原文。

    按句读切分后逐句比对归一词串的相似度，低于 0.4 视为无提示，
    避免把无关句子返回给模型造成误导。只用于拒绝回执的纠错指引，不构成写入依据。
    """
    quote_key, _ = _wordonly_scan(quote)
    if not quote_key:
        return None
    best: tuple[float, str] | None = None
    for match in re.finditer(r"[^。！？!?;；,，.\n]+", source_text):
        candidate = match.group().strip()
        candidate_key, _ = _wordonly_scan(candidate)
        if not candidate_key:
            continue
        ratio = difflib.SequenceMatcher(None, quote_key, candidate_key, autojunk=False).ratio()
        if ratio >= 0.4 and (best is None or ratio > best[0]):
            best = (ratio, candidate[:max_chars])
    return best[1] if best else None


@dataclass(frozen=True)
class AnchorResolution:
    """锚定解析结果：成功携带原文逐字子串，失败携带稳定理由码与纠错提示。"""

    quote: str = ""
    reason: str = ""
    hint: str = ""


_ANCHOR_MAX_SPAN_CHARS = 600


def _key_span_to_source(
    spans: list[tuple[int, int, int, int]], key_start: int, key_end: int
) -> tuple[int, int]:
    """把归一键区间映射回原文区间 [start, end)。"""
    start = end = -1
    for k_lo, k_hi, s_lo, s_hi in spans:
        if start < 0 and k_hi > key_start:
            start = s_lo
        if k_lo >= key_end:
            break
        end = s_hi
    return start, end


def anchor_resolve(
    source_text: str,
    quote_start: str,
    quote_end: str = "",
    max_span_chars: int = _ANCHOR_MAX_SPAN_CHARS,
) -> AnchorResolution:
    """按段落首尾片段在原文中定位，返回原文逐字子串；模型不手抄整段。

    匹配规则与写入判定一致（词序列归一，忽略空白/标点样式）。
    失败理由码：empty_anchor / anchor_not_found / ambiguous_start /
    end_not_found / span_too_long；not_found 附最近似原文句作提示。
    """
    source_key, spans = _wordonly_scan(source_text)
    start_key, _ = _wordonly_scan(quote_start)
    if not start_key:
        return AnchorResolution(reason="empty_anchor")
    occurrences: list[int] = []
    position = source_key.find(start_key)
    while position >= 0:
        occurrences.append(position)
        position = source_key.find(start_key, position + 1)
    if not occurrences:
        hint = nearest_source_passage(source_text, quote_start) or ""
        return AnchorResolution(reason="anchor_not_found", hint=hint)
    if len(occurrences) > 1:
        return AnchorResolution(reason="ambiguous_start")
    key_start = occurrences[0]
    key_end = key_start + len(start_key)
    if quote_end:
        end_key, _ = _wordonly_scan(quote_end)
        if not end_key:
            return AnchorResolution(reason="empty_anchor")
        end_position = source_key.find(end_key, key_end)
        if end_position < 0:
            return AnchorResolution(reason="end_not_found")
        key_end = max(key_end, end_position + len(end_key))
    source_start, source_stop = _key_span_to_source(spans, key_start, key_end)
    if source_start < 0 or source_stop <= source_start:
        return AnchorResolution(reason="anchor_not_found")
    quote = source_text[source_start:source_stop].strip()
    if not quote:
        return AnchorResolution(reason="anchor_not_found")
    if len(quote) > max_span_chars:
        return AnchorResolution(reason="span_too_long")
    return AnchorResolution(quote=quote)
