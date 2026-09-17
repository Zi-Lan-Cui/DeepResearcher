"""Evidence 的确定性来源校验：只保证"quote 逐字在原文里"。

不做行号定位——行号会引入 splitlines/`\n` 数行口径不一致、重复句错定位等脆弱性，
而它对 Writer/前端零消费（引用靠 [来源N]→URL）。唯一不变式：
`normalize_text(quote)` 是 `normalize_text(source_text)` 的子串。
模型能逐字复现某段文本，当且仅当它被展示过该文本，故无需额外的"读过区间"追踪。
"""

import re


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def quote_in_source(source_text: str, quote: str) -> bool:
    """quote 逐字内容必须出现在原文中，忽略一切空白差异（换行/多空格/换页）。

    用"去掉所有空白后子串"而非"折叠为单空格"：PDF/网页正文里 quote 跨越的位置可能是
    `\n`、`\f` 或多空格，模型复现时空白形态不确定；只要求非空白字符序列逐字一致。
    """
    stripped_quote = re.sub(r"\s+", "", quote).lower()
    if not stripped_quote:
        return False
    stripped_source = re.sub(r"\s+", "", source_text).lower()
    return stripped_quote in stripped_source
