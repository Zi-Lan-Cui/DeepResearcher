"""跨模块共享的字面量词汇表，不改变序列化后的 wire 值。"""

from typing import Literal

# Evidence/Citation/模型提交三处共用的支撑强度。
# strict 表示只接受原文；partial 允许页面抓取失败时降级使用搜索摘要。
Support = Literal["partial", "strict"]

# Supervisor 材料成熟度：综合稿能否支撑成文。
GenerationMode = Literal["not_ready", "partial", "full"]

# 研究阶段生命周期(WriterDirective 交接与 SupervisorProgress 共用)。
ResearchStatus = Literal["not_started", "running", "completed", "incomplete", "failed"]

# Writer 可交付的四种答案形态。
WriterAnswerMode = Literal["quick_answer", "deep_research", "research_incomplete", "review_limited"]
# State 顶层 answer_mode 通道：与 Writer 交付形态同集。等待澄清由 run 的运行状态
# awaiting_input 承载，不进交付形态词表。
AnswerMode = Literal[
    "quick_answer",
    "deep_research",
    "research_incomplete",
    "review_limited",
]

INCOMPLETE_REPORT_HEADING = "报告未生成"

# 研究材料来源：本地抓取、云端抓取、搜索接口随结果带回的正文与摘要。
RETRIEVAL_ORIGIN_FETCH = "origin_fetch"
RETRIEVAL_ALIYUN_WEB_FETCH = "aliyun_web_fetch"
RETRIEVAL_DEEPSEEK_WEB_FETCH = "deepseek_web_fetch"
RETRIEVAL_TAVILY_RAW_CONTENT = "tavily_raw_content"
RETRIEVAL_DEEPSEEK_SEARCH_CONTENT = "deepseek_search_content"
# 没有专属词的搜索供应商带回正文时用它，不冒用别家的名字。
RETRIEVAL_SEARCH_PROVIDER_CONTENT = "search_provider_content"
RETRIEVAL_SEARCH_SUMMARY = "search_summary"

# 证据支持强度的顺序与排名，便于比较大小。
SUPPORT_ORDER: tuple[str, ...] = ("partial", "strict")
SUPPORT_RANK: dict[str, int] = {value: index for index, value in enumerate(SUPPORT_ORDER)}
