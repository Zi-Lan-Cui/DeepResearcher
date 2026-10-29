"""网页抓取能力、Provider 与统一结果契约。"""

from deepresearcher.tools.web.fetch.models import SourceDocument, SourceReaderToolResult
from deepresearcher.tools.web.fetch.protocol import FetchProvider
from deepresearcher.tools.web.fetch.providers import (
    AliyunFetchProvider,
    DeepSeekFetchProvider,
    DirectHttpFetchProvider,
    build_fetch_providers,
)
from deepresearcher.tools.web.fetch.service import FetchService

__all__ = [
    "AliyunFetchProvider",
    "DeepSeekFetchProvider",
    "DirectHttpFetchProvider",
    "FetchProvider",
    "FetchService",
    "SourceDocument",
    "SourceReaderToolResult",
    "build_fetch_providers",
]
