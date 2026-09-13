"""网页证据召回、抽取和校验。"""

from typing import TYPE_CHECKING

from deepresearcher.evidence.models import Evidence, EvidenceExtraction
from deepresearcher.evidence.tokens import get_token_estimator

if TYPE_CHECKING:
    from deepresearcher.evidence.extractor import EvidenceExtractor

__all__ = ["Evidence", "EvidenceExtraction", "EvidenceExtractor", "get_token_estimator"]


def __getattr__(name: str):
    """EvidenceExtractor 依赖 tools 层；包初始化时急切导入会与 tools 形成循环。"""
    if name == "EvidenceExtractor":
        from deepresearcher.evidence.extractor import EvidenceExtractor

        return EvidenceExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
