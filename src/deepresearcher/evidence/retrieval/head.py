"""文档头部硬截断基线。"""

from deepresearcher.tools.web.parsing.models import DocumentBlock


class HeadTruncateRetriever:
    """按原文顺序返回全部块，由抽取器在 token 预算处硬截断。

    该后端只用作 BM25 的可重复 A/B 基线，不根据查询排序，
    也不扩展相邻块。
    """

    def select(
        self,
        blocks: list[DocumentBlock],
        query: str,
        top_k: int,
        window: int,
    ) -> list[DocumentBlock]:
        del query, top_k, window
        return list(blocks)
