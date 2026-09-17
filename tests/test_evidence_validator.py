from deepresearcher.evidence.validator import normalize_text, quote_in_source


def test_quote_in_source_accepts_verbatim_ignoring_whitespace():
    source = "跨行引用上半句\n   下半句结束，其余正文。"
    assert quote_in_source(source, "跨行引用上半句 下半句结束")
    assert quote_in_source(source, "跨行引用上半句下半句结束")  # 忽略换行/空白差异


def test_quote_in_source_rejects_unseen_or_empty_quote():
    assert not quote_in_source("已有正文", "虚构引用")
    assert not quote_in_source("已有正文", "")  # 空 quote 不算证据


def test_quote_in_source_is_immune_to_duplicate_and_paginated_lines():
    # 旧实现按行号定位，遇 \f 换页 / \r\n / 重复句会错位；纯子串判定不受影响。
    source = "同一句结论。\f同一句结论。\r\n正文其他内容。"
    assert quote_in_source(source, "同一句结论。")
    assert normalize_text("A  \n B") == "a b"
