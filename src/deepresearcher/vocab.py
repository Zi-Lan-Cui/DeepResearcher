"""跨模块共享的字面量词汇表:同一值集只登记一次,不改变序列化后的 wire 值。

必须住在根级而非 schemas/ 之内:evidence.models 也要 import 它,放进 schemas 包
会形成 evidence 与 schemas 的循环导入,模块在半初始化状态下就会报错。
本模块只依赖 typing。

"一个研究方向"在链上有三个名字,各司其职,不要再新增别名:
- ``topic``:Supervisor 派发工具 ResearchDelegate 的入参;
- ``task`` / ``task_id``:状态簿记,含 SubTask、事件字段、ResearchDirectionResult;
- ``direction``:对外与模型可见面,含事件名 direction_*、方向卡、提示词与报告用词。
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
# State 顶层 answer_mode 通道:与 Writer 交付形态同集。等待澄清由 run 的运行状态
# awaiting_input 承载,不进交付形态词表。
AnswerMode = Literal[
    "quick_answer",
    "deep_research",
    "research_incomplete",
    "review_limited",
]

# retrieval_method 词表:默认与回退共用 origin_fetch,每种来源路径各一个常量。
# Evidence 字段类型是 str:fetch 编排层的 f"{provider.name}_fetch" 兜底允许产生
# 词表外的新值,域核心因此不加枚举校验;provider 的原始名字不写入 Evidence。
RETRIEVAL_ORIGIN_FETCH = "origin_fetch"
RETRIEVAL_ALIYUN_WEB_FETCH = "aliyun_web_fetch"
RETRIEVAL_TAVILY_RAW_CONTENT = "tavily_raw_content"
RETRIEVAL_SEARCH_SUMMARY = "search_summary"

# support 强度从低到高的排序:writer 打分与 researcher 分档共用,各处不要再抄一份。
SUPPORT_ORDER: tuple[str, ...] = ("insufficient", "partial", "direct")
SUPPORT_RANK: dict[str, int] = {value: index for index, value in enumerate(SUPPORT_ORDER)}
