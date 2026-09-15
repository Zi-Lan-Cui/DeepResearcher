"""可替换的块级检索后端。"""

from typing import Literal

from deepresearcher.evidence.retrieval.base import BlockRetriever
from deepresearcher.evidence.retrieval.bm25 import BM25Retriever, select_blocks
from deepresearcher.evidence.retrieval.head import HeadTruncateRetriever


def default_block_retriever(
    backend: Literal["bm25", "head_truncate"] | str = "bm25",
) -> BlockRetriever:
    """构造默认检索器；未知后端在装配期立即失败。"""
    if backend == "bm25":
        return BM25Retriever()
    if backend == "head_truncate":
        return HeadTruncateRetriever()
    raise ValueError(f"不支持的 Evidence Retriever 后端：{backend}")


__all__ = [
    "BM25Retriever",
    "BlockRetriever",
    "HeadTruncateRetriever",
    "default_block_retriever",
    "select_blocks",
]
