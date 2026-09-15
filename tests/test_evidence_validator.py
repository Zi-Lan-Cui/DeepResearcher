import pytest

from deepresearcher.evidence.validator import locate_quote_lines


def test_locate_quote_lines_handles_multiline_and_blank_lines():
    text = "标题\n\n第一段开头\n跨行引用上半句\n  下半句结束\n尾声"

    assert locate_quote_lines(text, "跨行引用上半句 下半句结束") == (4, 5)


def test_locate_quote_lines_rejects_unseen_quote():
    with pytest.raises(ValueError, match="不存在"):
        locate_quote_lines("已有正文", "虚构引用")
