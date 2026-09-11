"""契约模型包：按域分文件，公共 import 路径在此聚合。

- ``reporting``  报告域：任务书、引用元数据、段落绑定
- ``sections``   State 分区与生命周期：进度模型、结果契约、StopReason 词汇
- ``tool_args``  工具调用入参与工具结果
- ``decisions``  模型结构化输出的决策 Schema

外部一律 ``from deepsearch_agent.schemas import X``，不直接 import 子模块，
拆分对调用方保持零改动。
"""

from deepsearch_agent.schemas.decisions import (
    ReflectionDecision,
    ResearchDirectionDecision,
    RouteDecision,
)
from deepsearch_agent.schemas.reporting import (
    Citation,
    CoveredTopic,
    MarkdownReportDraft,
    ParagraphBinding,
    ReportBrief,
    ResearchAspect,
    ResearchSynthesis,
    WriterDirective,
)
from deepsearch_agent.schemas.sections import (
    ResearchAgentResult,
    ResearchDirectionResult,
    ResearchProgress,
    ReviewIssue,
    ReviewProgress,
    RunError,
    RunLifecycle,
    StopReason,
    SupervisorStateUpdate,
    WriterProgress,
    WriterResult,
)
from deepsearch_agent.schemas.sources import SourceProfile
from deepsearch_agent.schemas.tool_args import (
    ReadSources,
    ReadWorkingSet,
    ReleaseEvidence,
    ResearchComplete,
    ResearchDelegate,
    ResearchDirectionComplete,
    ResearchToolResult,
    RestoreEvidence,
    ReviseResearchSynthesis,
    SearchSources,
)

__all__ = [
    "Citation",
    "CoveredTopic",
    "ReleaseEvidence",
    "MarkdownReportDraft",
    "ParagraphBinding",
    "ReadSources",
    "ReadWorkingSet",
    "ReflectionDecision",
    "ReportBrief",
    "ResearchAspect",
    "ResearchAgentResult",
    "ResearchComplete",
    "ResearchDelegate",
    "ResearchDirectionComplete",
    "ResearchDirectionDecision",
    "ResearchDirectionResult",
    "ResearchProgress",
    "ResearchSynthesis",
    "ResearchToolResult",
    "RestoreEvidence",
    "ReviseResearchSynthesis",
    "ReviewIssue",
    "ReviewProgress",
    "RouteDecision",
    "RunError",
    "RunLifecycle",
    "SearchSources",
    "StopReason",
    "SourceProfile",
    "SupervisorStateUpdate",
    "WriterDirective",
    "WriterProgress",
    "WriterResult",
]
