from deepresearcher.evidence.validator import (
    normalize_text,
    quote_in_source,
    quote_verbatim_strict,
)


def test_encoding_variants_rescued_by_loose_but_rejected_by_strict():
    # PDF 软连字符 / 断词连字符 / ligature：忠实引用被 strict 误杀，loose 救回。
    for source, quote in (
        ("exam­ple system", "example system"),  # 软连字符
        ("exam-\nple system", "example system"),  # 断词换行
        ("The ﬁnding is robust", "The finding is robust"),  # ligature ﬁ→fi
    ):
        assert quote_in_source(source, quote), (source, quote)
        assert not quote_verbatim_strict(source, quote), (source, quote)


def test_paraphrase_fails_both_strict_and_loose():
    source = "The system remained stable for 50 hours."
    assert quote_in_source(
        source, "The system remained stable for 50 hours"
    )  # 逐字(去句号空白)仍过
    assert not quote_in_source(source, "The system was stable for about two days")  # 改述不过


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
