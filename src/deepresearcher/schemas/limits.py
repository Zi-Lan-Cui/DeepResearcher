"""Schema 容量边界。

业务裁剪由 ``AgentConfig`` 和工具处理器负责；本模块中的 hard limit
只用于拦截异常膨胀的模型输出，必须明显高于正常运行配置。
少数 per-call 常量是明确的工具协议限制，用于控制单次外部请求成本。
"""

# 异常保护上限：不应被当作日常业务配置。
STRUCTURED_IDENTIFIER_HARD_LIMIT_CHARS = 512
# 错误文本是旁路诊断,不是载荷;进 checkpoint/DB/事件前钳到可读规模。
RUN_ERROR_TEXT_HARD_LIMIT_CHARS = 2_000
STRUCTURED_TEXT_HARD_LIMIT_CHARS = 16_000
STRUCTURED_SUMMARY_HARD_LIMIT_CHARS = 64_000
STRUCTURED_COLLECTION_HARD_LIMIT = 128
EVIDENCE_REFERENCES_HARD_LIMIT = 256
REPORT_MARKDOWN_HARD_LIMIT_CHARS = 256_000
REPORT_CAVEATS_HARD_LIMIT = 50

# Researcher 单次工具调用协议：限制一次外部操作的扇出，不限制整个方向。
SEARCH_QUERIES_PER_CALL = 2
SOURCE_CANDIDATES_PER_READ = 8
SEARCH_RESULTS_PAGE_HARD_LIMIT = 20
SEARCH_RESULTS_PREVIEW_COUNT = 10
SEARCH_RESULT_SNIPPET_PREVIEW_CHARS = 1_200
# 仅用于 search 客户端审计事件(日志排查用):独立于模型预览数的展示条数,
# 标题按字符截断。二者都不进模型上下文,改这里只影响日志可读性。
SEARCH_RESULTS_AUDIT_PREVIEW_COUNT = 8
SEARCH_RESULT_TITLE_PREVIEW_CHARS = 160
# 回合思维链 content_preview 的字符预算:observability 产生端截一次、projector
# 消费端(deny-by-default,不信任上游)再钳一次,两处共用此常量。
EVENT_CONTENT_PREVIEW_CHARS = 800
# 历史标题(clarify 浓缩问题)与报告标题(writer 命名回答)的长度上限。
RUN_HEADLINE_MAX_CHARS = 40
REPORT_TITLE_MAX_CHARS = 80

# Clarifier 交互协议边界(原 agents/clarifier/constants.py,wire 容量在本模块登记)。
MAX_CLARIFICATION_ROUNDS = 2
CLARIFICATION_QUESTION_MAX_CHARS = 500
CLARIFICATION_OPTION_COUNT = 3
CLARIFICATION_INTENT_MAX_CHARS = 1_000
CLARIFICATION_FOCUS_LIMIT = 4
CLARIFICATION_ASSUMPTION_LIMIT = 3
# ReadEvidence 一次可请求的窗口(防失控的宽松值);每轮实际交付量由
# writer_read_batch_size 决定,差额走 not_read_ids 显式排队。
READ_EVIDENCE_REQUEST_WINDOW_IDS = 50
# 方向卡短题(ResearchDelegate 派发时给出的展示名)上限。
DIRECTION_TITLE_MAX_CHARS = 40
