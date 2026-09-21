"""契约模型包：按域分文件，公共 import 路径在此聚合。

- ``reporting``  报告域：任务书、引用元数据、段落绑定
- ``sections``   State 分区与生命周期：进度模型、结果契约、StopReason 词汇
- ``tool_args``  工具调用入参与工具结果
- ``decisions``  模型结构化输出的决策 Schema

外部一律 ``from deepresearcher.schemas import X``，不直接 import 子模块，
拆分对调用方保持零改动。
"""

from deepresearcher.schemas.decisions import (
    ResearchDirectionDecision,
    ReviewDecision,
    RouteDecision,
)
from deepresearcher.schemas.reporting import (
    Citation,
    CoveredTopic,
    MarkdownReportDraft,
    ParagraphBinding,
    ReportBrief,
    ResearchAspect,
    ResearchSynthesis,
    WriterDirective,
)
from deepresearcher.schemas.sections import (
    DirectionStopReason,
    RenderOutcome,
    ResearchAgentResult,
    ResearchDirectionResult,
    ReviewIssue,
    ReviewProgress,
    RunError,
    RunStatus,
    StopReason,
    SupervisorProgress,
    SupervisorStateUpdate,
    WriterProgress,
    WriterResult,
    terminal_reason_text,
)
from deepresearcher.schemas.sources import SourceProfile
from deepresearcher.schemas.tool_args import (
    TOOL_RECEIPT_PREFIX,
    AddEvidence,
    DocumentLineRange,
    EvidenceSubmission,
    GrepDocument,
    ListSearchResults,
    ReadDocument,
    ReadSources,
    ReadWorkingSet,
    ReleaseEvidence,
    ResearchComplete,
    ResearchDelegate,
    ResearchDirectionComplete,
    RestoreEvidence,
    ReviseResearchSynthesis,
    SearchSources,
    format_tool_receipt,
)

__all__ = [
    "AddEvidence",
    "format_tool_receipt",
    "TOOL_RECEIPT_PREFIX",
    "Citation",
    "CoveredTopic",
    "DocumentLineRange",
    "EvidenceSubmission",
    "GrepDocument",
    "ListSearchResults",
    "ReleaseEvidence",
    "MarkdownReportDraft",
    "ParagraphBinding",
    "ReadSources",
    "ReadDocument",
    "ReadWorkingSet",
    "ReviewDecision",
    "RenderOutcome",
    "ReportBrief",
    "ResearchAspect",
    "ResearchAgentResult",
    "ResearchComplete",
    "ResearchDelegate",
    "ResearchDirectionComplete",
    "ResearchDirectionDecision",
    "ResearchDirectionResult",
    "SupervisorProgress",
    "ResearchSynthesis",
    "RestoreEvidence",
    "ReviseResearchSynthesis",
    "ReviewIssue",
    "ReviewProgress",
    "RouteDecision",
    "RunError",
    "RunStatus",
    "SearchSources",
    "DirectionStopReason",
    "StopReason",
    "terminal_reason_text",
    "SourceProfile",
    "SupervisorStateUpdate",
    "WriterDirective",
    "WriterProgress",
    "WriterResult",
]
