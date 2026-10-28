"""Evidence 的确定性来源校验：按首尾片段锚在原文中定位"""

import bisect
import difflib
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

_ANCHOR_MAX_SPAN_CHARS = 600
# 锚定片段过长时，模型可能会截断或改写，导致无法在原文中找到对应位置。
_SENTENCE_BREAKS = frozenset("。！？!?;；,，.\n")
# 低于此相似度不返回提示，避免把无关句子交给模型。
_NEAREST_MIN_RATIO = 0.4


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


@dataclass(frozen=True)
class AnchorResolution:
    """锚定解析结果：成功携带原文逐字子串，失败携带稳定理由码与纠错提示。"""

    quote: str = ""
    reason: str = ""
    hint: str = ""


@dataclass(frozen=True)
class SourceIndex:
    """一份原文的归一键串与键到源区间映射，同一原文的多条锚点共用一次扫描。"""

    text: str
    key: str
    spans: tuple[tuple[int, int, int, int], ...]
    sentences: tuple[tuple[int, int, int, int], ...]

    @classmethod
    def scan(cls, text: str) -> "SourceIndex":
        """归一全文，同时记下逐字符映射与句子边界。"""
        key, spans, sentences = _wordonly_scan(text)
        return cls(text=text, key=key, spans=tuple(spans), sentences=tuple(sentences))

    def resolve(
        self,
        quote_start: str,
        quote_end: str = "",
        max_span_chars: int = _ANCHOR_MAX_SPAN_CHARS,
    ) -> AnchorResolution:
        """按段落首尾片段在原文中定位，返回原文逐字子串。"""
        start_key, _, _ = _wordonly_scan(quote_start)
        if not start_key:
            return AnchorResolution(reason="empty_anchor")
        occurrences: list[int] = []
        position = self.key.find(start_key)
        while position >= 0:
            occurrences.append(position)
            position = self.key.find(start_key, position + 1)
        if not occurrences:
            return AnchorResolution(reason="anchor_not_found")
        if len(occurrences) > 1:
            return AnchorResolution(reason="ambiguous_start")
        key_start = occurrences[0]
        key_end = key_start + len(start_key)
        if quote_end:
            end_key, _, _ = _wordonly_scan(quote_end)
            if not end_key:
                return AnchorResolution(reason="empty_anchor")
            end_position = self.key.find(end_key, key_end)
            if end_position < 0:
                return AnchorResolution(reason="end_not_found")
            key_end = max(key_end, end_position + len(end_key))
        source_start, source_stop = _key_span_to_source(self.spans, key_start, key_end)
        if source_start < 0 or source_stop <= source_start:
            return AnchorResolution(reason="anchor_not_found")
        quote = self.text[source_start:source_stop].strip()
        if not quote:
            return AnchorResolution(reason="anchor_not_found")
        if len(quote) > max_span_chars:
            return AnchorResolution(reason="span_too_long")
        return AnchorResolution(quote=quote)

    def nearest_passage(self, quote: str, max_chars: int = 200) -> str | None:
        """返回与改述引用词面重叠最高的原文句子，供模型替换成正确的原文。"""
        quote_key, _, _ = _wordonly_scan(quote)
        quote_len = len(quote_key)
        if not quote_len:
            return None
        best: tuple[float, str] | None = None
        for key_lo, key_hi, source_lo, source_hi in self.sentences:
            candidate_len = key_hi - key_lo
            # 相似度上界是 2*短/总长，故短的一方短于长的一方的四分之一时必然不达标，
            # 先用整数比较排掉，省下逐句构造 SequenceMatcher 的开销。
            if min(quote_len, candidate_len) * 4 < max(quote_len, candidate_len):
                continue
            matcher = difflib.SequenceMatcher(
                None, quote_key, self.key[key_lo:key_hi], autojunk=False
            )
            ratio = matcher.ratio()
            if ratio >= _NEAREST_MIN_RATIO and (best is None or ratio > best[0]):
                best = (ratio, self.text[source_lo:source_hi][:max_chars])
        return best[1] if best else None


def anchor_resolve(
    source_text: str,
    quote_start: str,
    quote_end: str = "",
    max_span_chars: int = _ANCHOR_MAX_SPAN_CHARS,
) -> AnchorResolution:
    """单条锚点的便捷入口，未命中时附最近似的原文句；批量场景直接用 SourceIndex。"""
    index = SourceIndex.scan(source_text)
    resolution = index.resolve(quote_start, quote_end, max_span_chars)
    if resolution.reason != "anchor_not_found":
        return resolution
    return AnchorResolution(reason=resolution.reason, hint=index.nearest_passage(quote_start) or "")


def nearest_source_passage(source_text: str, quote: str, max_chars: int = 200) -> str | None:
    """单条改述引用的便捷入口；同上。"""
    return SourceIndex.scan(source_text).nearest_passage(quote, max_chars)


def _is_word_char(char: str) -> bool:
    return char.isalnum() or "一" <= char <= "鿿"


def _wordonly_scan(
    text: str,
) -> tuple[str, list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    """归一全文（NFKC + casefold，只留字母数字与 CJK），同时记下键到源映射与句子边界。"""
    pieces: list[str] = []
    spans: list[tuple[int, int, int, int]] = []
    sentences: list[tuple[int, int, int, int]] = []
    cursor = 0
    sentence_key_start = 0
    sentence_source_start = -1
    for position, char in enumerate(text):
        if char in _SENTENCE_BREAKS:
            if sentence_source_start >= 0:
                sentences.append((sentence_key_start, cursor, sentence_source_start, position))
                sentence_source_start = -1
            sentence_key_start = cursor
            continue
        fragment = unicodedata.normalize("NFKC", char).casefold()
        kept = "".join(ch for ch in fragment if _is_word_char(ch))
        if not kept:
            continue
        if sentence_source_start < 0:
            sentence_source_start = position
        pieces.append(kept)
        spans.append((cursor, cursor + len(kept), position, position + 1))
        cursor += len(kept)
    if sentence_source_start >= 0:
        sentences.append((sentence_key_start, cursor, sentence_source_start, len(text)))
    return "".join(pieces), spans, sentences


def _key_span_to_source(
    spans: Sequence[tuple[int, int, int, int]], key_start: int, key_end: int
) -> tuple[int, int]:
    """把归一键区间映射回原文区间 [start, end)；键区间单调，二分定位而不再扫全文。"""
    first = bisect.bisect_right(spans, key_start, key=lambda span: span[1])
    if first >= len(spans):
        return -1, -1
    last = bisect.bisect_left(spans, key_end, key=lambda span: span[0]) - 1
    if last < first:
        return -1, -1
    return spans[first][2], spans[last][3]
