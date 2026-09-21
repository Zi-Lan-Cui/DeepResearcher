"""跨模块共享的字面量词汇表:同一值集只登记一次,wire 形状不变。

必须住在根级而非 schemas/ 之内:evidence.models 也要 import 它,放进
schemas 包会重新点燃 evidence⇄schemas 的半初始化引信。本模块只依赖 typing。

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

# Evidence 来源方法默认值的公开常量:Literal 默认/回退/拼接六处共用此一词。
RETRIEVAL_ORIGIN_FETCH = "origin_fetch"

# support 阶梯的序(升序):writer 打分与 researcher 分档共用,不再各写形状。
SUPPORT_ORDER: tuple[str, ...] = ("insufficient", "partial", "direct")
SUPPORT_RANK: dict[str, int] = {value: index for index, value in enumerate(SUPPORT_ORDER)}
