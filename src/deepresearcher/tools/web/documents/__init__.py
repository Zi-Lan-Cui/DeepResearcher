"""Researcher 可控读取的持久文档存储。"""

from deepresearcher.tools.web.documents.local import LocalDocumentStore
from deepresearcher.tools.web.documents.models import (
    DocumentGrepMatch,
    DocumentOutlineItem,
    DocumentReadRange,
    DocumentRef,
    DocumentView,
)
from deepresearcher.tools.web.documents.store import DocumentStore

__all__ = [
    "DocumentGrepMatch",
    "DocumentOutlineItem",
    "DocumentReadRange",
    "DocumentRef",
    "DocumentStore",
    "DocumentView",
    "LocalDocumentStore",
]
