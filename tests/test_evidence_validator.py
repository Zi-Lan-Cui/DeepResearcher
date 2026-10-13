from deepresearcher.evidence.validator import (
    anchor_resolve,
    collapse_whitespace,
    nearest_source_passage,
)


def test_anchor_pair_resolves_verbatim_source_substring():
    # 首尾锚之间整段由程序逐字回填，弯引号与破折号原样保留。
    source = "引言。The report — “very robust” — passed. 结尾"
    result = anchor_resolve(source, "The report", "passed")
    assert result.reason == ""
    assert result.quote == "The report — “very robust” — passed"
    assert result.quote in source  # 产物必须是原文严格子串


def test_anchor_ignores_punctuation_style_without_repair_tier():
    # 锚归一只比词序列：模型用直引号抄写也能命中弯引号原文，回填的是原文子串。
    source = "报告称“非常稳健”，通过验收。其余正文。"
    result = anchor_resolve(source, "报告称'非常稳健',通过验收")
    assert result.reason == ""
    assert result.quote == "报告称“非常稳健”，通过验收"


def test_start_anchor_alone_yields_its_own_fragment():
    source = "正文。The system remained stable for 50 hours. 其余"
    result = anchor_resolve(source, "The system remained stable for 50 hours")
    assert result.reason == ""
    assert result.quote == "The system remained stable for 50 hours"


def test_anchor_rescues_encoding_variants():
    # PDF 软连字符 / 断词换行 / ligature：归一为词序列后天然命中。
    expected = {
        "exam­ple system": "exam­ple",
        "exam-\nple system": "exam-\nple",
        "The ﬁnding is robust": "The ﬁnding",
    }
    for source, quote in expected.items():
        result = anchor_resolve(source, "example" if "exam" in source else "the finding")
        assert result.reason == "", (source, result.reason)
        assert result.quote == quote, (source, result.quote)


def test_ambiguous_start_anchor_is_rejected():
    # 重复句里短锚命中多处：错误必须暴露为歧义，而不是任选一处静默入池。
    source = "同一句结论。\f同一句结论。\r\n正文其他内容。"
    result = anchor_resolve(source, "同一句结论")
    assert result.reason == "ambiguous_start"


def test_end_anchor_must_follow_start_anchor():
    source = "第一段是结论。\n第二段提到方法与数据来源。"
    result = anchor_resolve(source, "第二段", "第一段")
    assert result.reason == "end_not_found"


def test_punctuation_only_anchor_is_empty_anchor():
    result = anchor_resolve("已有正文", "—…—")
    assert result.reason == "empty_anchor"


def test_not_found_anchor_carries_nearest_sentence_hint():
    source = "The system remained stable for 50 hours. Another matter entirely here."
    result = anchor_resolve(source, "The device stayed up for two days")
    assert result.reason == "anchor_not_found"
    assert result.hint == "The system remained stable for 50 hours"


def test_span_over_limit_is_rejected():
    source = "开头词 " + "中" * 700 + " 结尾词"
    result = anchor_resolve(source, "开头词", "结尾词")
    assert result.reason == "span_too_long"


def test_nearest_passage_points_at_the_overlapping_sentence():
    source = "The system remained stable for 50 hours. Another matter entirely here."
    hint = nearest_source_passage(source, "The system stayed stable for two days")
    assert hint == "The system remained stable for 50 hours"


def test_nearest_passage_returns_none_when_nothing_is_similar():
    source = "完全不同的一句话。Another matter entirely."
    assert nearest_source_passage(source, "quantum tunneling in semiconductors") is None


def test_collapse_whitespace_normalizes_run():
    assert collapse_whitespace("A  \n B") == "a b"
