"""Evidence 的确定性来源校验：只保证 quote 的非空白字符序列逐字出现在原文里。

不做行号定位：行号引入 splitlines 计数口径不一致、重复句定位错误等脆弱性，
而 Writer 与前端不消费行号，引用通过 [证据N]→URL 完成。

两级判定加一档修复：
- quote_verbatim_strict：仅忽略空白差异，比入池判定更严格，供调用方区分拒绝原因。
- quote_in_source（入池判定）：在忽略空白之上再做 NFKC 归一并去除软连字符与断词
  连字符，使 PDF、网页里 exam‑ple、exam-\nple、ﬁ 这类编码变体的忠实引用仍被接受。
  两侧对称归一，改述会改变字母序列，因此无法匹配。
- quote_verbatim_span（修复档）：入池判定不通过时，若忽略标点后词序列一致，
  回退定位到原文对应子串。拒绝记录中一半以上只差引号或破折号样式，
  回退让这部分直接入池，模型不必整轮重新抄写。
调用方据此区分编码导致的误拒：strict 不通过但入池判定通过，以及模型改述：
入池判定也不通过。
"""

import difflib
import re
import unicodedata

_SOFT_HYPHENS = "­‐‑"  # 软连字符、连字符、不换行连字符


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _norm_whitespace(text: str) -> str:
    """去除全部空白并转小写，对应严格判定使用的口径。"""
    return re.sub(r"\s+", "", str(text)).lower()


def _norm_loose(text: str) -> str:
    """NFKC 归一 + 去软连字符 + 去所有连字符与空白 + 小写。"""
    normalized = unicodedata.normalize("NFKC", str(text))
    for ch in _SOFT_HYPHENS:
        normalized = normalized.replace(ch, "")
    return re.sub(r"[-\s]+", "", normalized).lower()


def quote_verbatim_strict(source_text: str, quote: str) -> bool:
    key = _norm_whitespace(quote)
    return bool(key) and key in _norm_whitespace(source_text)


def quote_in_source(source_text: str, quote: str) -> bool:
    """入池判定：忽略空白与连字符/ligature 后的忠实逐字。"""
    key = _norm_loose(quote)
    return bool(key) and key in _norm_loose(source_text)


def _is_word_char(char: str) -> bool:
    return char.isalnum() or "一" <= char <= "鿿"


def _wordonly_scan(text: str) -> tuple[str, list[tuple[int, int, int, int]]]:
    """只保留字母/数字/CJK 做归一（NFKC + casefold，剔标点/引号/破折号/省略号/零宽），
    并行返回逐字符的键区间映射 [(键起, 键止, 源起, 源止)]，供命中后回取原文子串。"""
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


def quote_verbatim_span(source_text: str, quote: str) -> str | None:
    """词序列一致、仅标点、引号、破折号或空白不同时，返回原文中对应的逐字子串。

    命中说明该引用是格式变体：模型抄写的词全部正确，只是标点样式不同。
    返回值是原文的严格子串，下游逐字校验将其视为直接引用；
    词序列不一致即改述时返回 None。
    """
    source_key, spans = _wordonly_scan(source_text)
    quote_key, _ = _wordonly_scan(quote)
    if not quote_key:
        return None
    position = source_key.find(quote_key)
    if position < 0:
        return None
    end_key = position + len(quote_key)
    start = end = -1
    for key_start, key_end, source_start, source_end in spans:
        if start < 0 and key_end > position:
            start = source_start
        if key_start >= end_key:
            break
        end = source_end
    if start < 0:
        return None
    return source_text[start:end]


def nearest_source_passage(source_text: str, quote: str, max_chars: int = 200) -> str | None:
    """返回与改述引用词面重叠最高的原文句子，供模型逐字替换为正确的原文。

    按句读切分后逐句比对归一词串的相似度，低于 0.4 视为无提示，
    避免把无关句子返回给模型造成误导。只用于拒绝回执的纠错指引，不构成入池依据。
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
