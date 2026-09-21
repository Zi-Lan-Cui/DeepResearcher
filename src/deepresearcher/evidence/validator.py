"""Evidence 的确定性来源校验：只保证"quote 逐字在原文里"。

不做行号定位——行号会引入 splitlines/`\n` 数行口径不一致、重复句错定位等脆弱性，
而它对 Writer/前端零消费（引用靠 [来源N]→URL）。唯一不变式：quote 的非空白字符序列
必须逐字出现在原文里。

两级匹配：
- `quote_verbatim_strict`：仅忽略空白差异（旧口径）。
- `quote_in_source`（入池判定）：在忽略空白之上，再做 NFKC + 去软连字符/断词连字符，
  把 PDF/网页里 `exam‑ple`、`exam-\nple`、ligature `ﬁ` 这类**忠实引用的编码变体**
  救回来。两侧对称归一，改述仍不可能匹配（字母序列不同）。
调用方据此区分"编码误杀"（strict 不过但 loose 过）与"模型改述"（loose 也不过）。
"""

import re
import unicodedata

_SOFT_HYPHENS = "­‐‑"  # soft hyphen / hyphen / no-break hyphen


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _norm_whitespace(text: str) -> str:
    """仅去除空白、小写（旧严格口径）。"""
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


def _norm_wordonly(text: str) -> str:
    """再进一步：NFKC + 只保留字母/数字/CJK，剔除一切标点、引号、破折号族、
    省略号、零宽与符号。用于区分"只差标点/引号样式（格式变体）"与"真的改词（改述）"。"""
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    return "".join(ch for ch in normalized if ch.isalnum() or "一" <= ch <= "鿿")


def quote_matches_ignoring_punctuation(source_text: str, quote: str) -> bool:
    """词序列一致、仅标点/引号/破折号/空白不同 → 判为格式变体（不是改述）。"""
    key = _norm_wordonly(quote)
    return bool(key) and key in _norm_wordonly(source_text)
