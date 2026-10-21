"""跨模块共享的字面量词汇表，不改变序列化后的 wire 值。

放在根级而非 schemas/ 内，避免 evidence 与 schemas 循环导入；本模块只依赖 typing。

同一个「研究方向」有三个名字，不再新增别名：
topic       ResearchDelegate 的入参
task        状态与事件字段，含 SubTask、ResearchDirectionResult
direction   对外与模型可见，含 direction_* 事件名、方向卡、提示词与报告用词
"""

from typing import Literal

# Evidence/Citation/模型提交三处共用的支撑强度。
Support = Literal["direct", "partial", "insufficient"]

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

# retrieval_method 词表：默认与回退共用 origin_fetch，每种来源路径各一个常量。
# 该字段是自由字符串：fetch 编排在没有已知来源路径时会用 f"{provider.name}_fetch"
# 作回退值，因此不做枚举校验。provider 原名不写入 Evidence。
RETRIEVAL_ORIGIN_FETCH = "origin_fetch"
RETRIEVAL_ALIYUN_WEB_FETCH = "aliyun_web_fetch"
RETRIEVAL_TAVILY_RAW_CONTENT = "tavily_raw_content"
RETRIEVAL_SEARCH_SUMMARY = "search_summary"

# support 强度从低到高的排序：writer 打分与 researcher 分档共用，各处不要再抄一份。
SUPPORT_ORDER: tuple[str, ...] = ("insufficient", "partial", "direct")
SUPPORT_RANK: dict[str, int] = {value: index for index, value in enumerate(SUPPORT_ORDER)}
