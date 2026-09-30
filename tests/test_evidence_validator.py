from deepresearcher.evidence.validator import (
    collapse_whitespace,
    nearest_source_passage,
    quote_in_source,
    quote_verbatim_span,
    quote_verbatim_strict,
)


def test_punctuation_only_diff_repairs_to_source_span():
    # 源用弯引号/破折号，模型写成直引号/连字符：词序列一致 → 回退定位到原文子串。
    source = "引言。The report — “very robust” — passed. 结尾"
    quote = 'The report - "very robust" - passed.'
    assert not quote_in_source(source, quote)  # 连字符/引号样式差异，第二档仍不过
    assert quote_verbatim_span(source, quote) == "The report — “very robust” — passed"


def test_verbatim_span_returns_strict_source_substring():
    # 修复产物必须过最严逐字口径，下游校验视同直接引用。
    source = "报告称“非常稳健”，通过验收。其余正文。"
    repaired = quote_verbatim_span(source, "报告称'非常稳健',通过验收")
    assert repaired == "报告称“非常稳健”，通过验收"
    assert quote_verbatim_strict(source, repaired)


def test_verbatim_span_rejects_paraphrase():
    source = "The system remained stable for 50 hours."
    assert quote_verbatim_span(source, "The device stayed up two days") is None


def test_verbatim_span_rejects_empty_or_punctuation_only_quote():
    assert quote_verbatim_span("已有正文", "") is None
    assert quote_verbatim_span("已有正文", "—…—") is None  # 归一后为空串不算命中


def test_nearest_passage_points_at_the_overlapping_sentence():
    source = "The system remained stable for 50 hours. Another matter entirely here."
    hint = nearest_source_passage(source, "The system stayed stable for two days")
    assert hint == "The system remained stable for 50 hours"


def test_nearest_passage_returns_none_when_nothing_is_similar():
    source = "完全不同的一句话。Another matter entirely."
    assert nearest_source_passage(source, "quantum tunneling in semiconductors") is None


def test_reworded_quote_is_paraphrase_across_all_tiers():
    source = "The system remained stable for 50 hours."
    assert quote_verbatim_span(source, "The device stayed up two days") is None
    assert not quote_in_source(source, "The device stayed up two days")


def test_encoding_variants_rescued_by_loose_but_rejected_by_strict():
    # PDF 软连字符 / 断词连字符 / ligature：忠实引用被 strict 错误拒绝、被 loose 接受。
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
    assert collapse_whitespace("A  \n B") == "a b"
