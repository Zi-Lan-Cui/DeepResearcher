"""不依赖 Embedding 的文档片段召回。"""

import re

from rank_bm25 import BM25Okapi

from deepsearch_agent.parsers.models import DocumentBlock


def _terms(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text.lower())


def select_blocks(
    query: str,
    blocks: list[DocumentBlock],
    *,
    top_k: int = 5,
    window: int = 1,
) -> list[DocumentBlock]:
    """用 BM25 筛选段落，并保留相邻块，避免切断限定条件。"""
    if not blocks:
        return []
    query_terms = _terms(query)
    corpus = []
    for block in blocks:
        heading_terms = _terms(" ".join(block.get("heading_path", [])))
        text_terms = _terms(block.get("text", ""))
        # 标题重复一次，相当于给标题路径增加权重。
        corpus.append(heading_terms + heading_terms + text_terms)

    scores = BM25Okapi(corpus).get_scores(query_terms)
    ranked = sorted(
        enumerate(scores),
        key=lambda item: item[1],
        reverse=True,
    )[:top_k]
    if not ranked or ranked[0][1] <= 0:
        return blocks[:top_k]
    selected: dict[int, DocumentBlock] = {}
    for index, _ in ranked:
        for neighbour in range(max(0, index - window), min(len(blocks), index + window + 1)):
            selected[neighbour] = blocks[neighbour]
    return [selected[index] for index in sorted(selected)]
