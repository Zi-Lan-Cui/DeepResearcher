"""BM25 块级检索实现。"""

import re

from rank_bm25 import BM25Okapi

from deepresearcher.tools.web.parsing.models import DocumentBlock


def _terms(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text.lower())


def _scores(query: str, blocks: list[DocumentBlock]) -> list[float]:
    query_terms = _terms(query)
    corpus: list[list[str]] = []
    for block in blocks:
        heading_terms = _terms(" ".join(block.get("heading_path", [])))
        text_terms = _terms(block.get("text", ""))
        # 标题重复一次，相当于给标题路径增加权重。
        corpus.append(heading_terms + heading_terms + text_terms)
    return list(BM25Okapi(corpus).get_scores(query_terms))


def _selected_indexes(
    query: str,
    blocks: list[DocumentBlock],
    *,
    top_k: int,
    window: int,
) -> tuple[set[int], list[float]]:
    if not blocks:
        return set(), []
    scores = _scores(query, blocks)
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]
    if not ranked or ranked[0][1] <= 0:
        return set(range(min(top_k, len(blocks)))), scores
    selected: set[int] = set()
    for index, _score in ranked:
        selected.update(range(max(0, index - window), min(len(blocks), index + window + 1)))
    return selected, scores


def select_blocks(
    query: str,
    blocks: list[DocumentBlock],
    *,
    top_k: int = 5,
    window: int = 1,
) -> list[DocumentBlock]:
    """用 BM25 筛选段落，并按文档顺序保留相邻块。"""
    selected, _scores_by_index = _selected_indexes(query, blocks, top_k=top_k, window=window)
    return [blocks[index] for index in sorted(selected)]


class BM25Retriever:
    """包装现有 BM25 召回，并以相关性优先顺序实现统一协议。"""

    def select(
        self,
        blocks: list[DocumentBlock],
        query: str,
        top_k: int,
        window: int,
    ) -> list[DocumentBlock]:
        selected, scores = _selected_indexes(query, blocks, top_k=top_k, window=window)
        # 分数相同时保持文档顺序；零分退化路径因此稳定返回文档前 top_k 块。
        ranked_indexes = sorted(selected, key=lambda index: (-scores[index], index))
        return [blocks[index] for index in ranked_indexes]
