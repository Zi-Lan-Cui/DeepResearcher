"""研究过程中可丢弃、可重建的搜索与正文材料。"""

from deepresearcher.tools.web.materials.models import SearchResultSet, StoredDocument
from deepresearcher.tools.web.materials.store import ResearchMaterialStore

__all__ = [
    "ResearchMaterialStore",
    "SearchResultSet",
    "StoredDocument",
]
