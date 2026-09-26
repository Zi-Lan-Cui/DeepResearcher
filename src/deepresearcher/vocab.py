"""跨模块共享的字面量词汇表:同一值集只登记一次,wire 形状不变。

必须住在根级而非 schemas/ 之内:evidence.models 也要 import 它;放进 schemas 包
会重新引入 evidence⇄schemas 循环导入,该包在半初始化状态下直接报错。本模块只依赖 typing。

两组刻意相近但**不同域**的词汇,读码时勿混:
- ``RouteDecision.route`` 的 ``clarify_needed``:路由节点的即时判定(wire 值已冻结);
- ``AnswerMode`` 的 ``clarification_needed``:run 的最终交付形态。
一字之差,语义空间互不包含,故不合并。
"""

from typing import Literal

# Evidence/Citation/模型提交三处共用的支撑强度。
Support = Literal["direct", "partial", "insufficient"]

# Supervisor 材料成熟度:综合稿能否支撑成文。
GenerationMode = Literal["not_ready", "partial", "full"]

# 研究阶段生命周期(WriterDirective 交接与 SupervisorProgress 共用)。
ResearchStatus = Literal["not_started", "running", "completed", "incomplete", "failed"]

# Writer 可交付的四种答案形态。
WriterAnswerMode = Literal["quick_answer", "deep_research", "research_incomplete", "review_limited"]
# State 顶层 answer_mode 通道 = Writer 产物 ∪ 路由侧澄清。
AnswerMode = Literal[
    "quick_answer",
    "deep_research",
    "research_incomplete",
    "review_limited",
    "clarification_needed",
]

# retrieval_method 词表:默认/回退共用 origin_fetch,各来源路径一名一常量。
# Evidence 字段是 str——fetch 编排层的 f"{provider.name}_fetch" 兜底允许产生
# 词表外的新值,域核心不设枚举闸(供应商字符串不进 Evidence)。
RETRIEVAL_ORIGIN_FETCH = "origin_fetch"
RETRIEVAL_ALIYUN_WEB_FETCH = "aliyun_web_fetch"
RETRIEVAL_TAVILY_RAW_CONTENT = "tavily_raw_content"
RETRIEVAL_SEARCH_SUMMARY = "search_summary"

# support 阶梯的序(升序):writer 打分与 researcher 分档共用,不再各写形状。
SUPPORT_ORDER: tuple[str, ...] = ("insufficient", "partial", "direct")
SUPPORT_RANK: dict[str, int] = {value: index for index, value in enumerate(SUPPORT_ORDER)}

# 审计事件 component 的归因域词表(谁产生的事件);工具名→归因的映射在
# observability/events/models 的组件表。用量记账是另一词表(category:
# search/fetch/llm,namespace: material_*),两域刻意不混——一边答"谁",一边答"花在哪"。
AuditComponent = Literal[
    "search_tool", "source_reader", "research_agent", "supervisor", "writer", "clarifier"
]
