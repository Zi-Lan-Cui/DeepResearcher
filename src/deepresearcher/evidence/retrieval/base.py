"""证据抽取前的块级检索协议。"""

from typing import Protocol

from deepresearcher.tools.web.parsing.models import DocumentBlock


class BlockRetriever(Protocol):
    """块级检索器抽象，供证据抽取预筛使用。

    返回顺序表示检索优先级；抽取器会在预算裁剪后恢复文档顺序。
    后续向量后端需实现相同协议，并在语义变化时同步升级
    ``chunking_version``，避免复用旧检索结果的 Evidence 缓存。
    """

    def select(
        self,
        blocks: list[DocumentBlock],
        query: str,
        top_k: int,
        window: int,
    ) -> list[DocumentBlock]: ...
