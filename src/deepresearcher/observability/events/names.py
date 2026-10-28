"""事件日志事件名的登记表，跨层 wire 契约。

发射侧与消费侧都引用这里的符号，改名只改这一处。动态拼接的名字按约定记录、
不作枚举成员：node_{status}、tool_{status}、{slug}_model_turn、{slug}_tool_{status}。
"""

from enum import StrEnum


class EventName(StrEnum):
    NODE_STARTED = "node_started"
    NODE_COMPLETED = "node_completed"
    NODE_FAILED = "node_failed"
    NODE_CANCELLED = "node_cancelled"
    SEARCH_QUERY_STARTED = "search_query_started"
    SEARCH_QUERY_COMPLETED = "search_query_completed"
    SEARCH_QUERY_FAILED = "search_query_failed"
    SOURCE_FETCH_STARTED = "source_fetch_started"
    SOURCE_FETCH_COMPLETED = "source_fetch_completed"
    SOURCE_DOCUMENT_REGISTERED = "source_document_registered"
    SOURCE_READ_FAILED = "source_read_failed"
    SOURCE_READ_SKIPPED = "source_read_skipped"
    # 抓取失败但用搜索提供方已带回的原文/摘要完成登记时的审计事件；
    # projector 不在白名单，对用户界面静默，仅落事件日志供归因。
    SOURCE_READ_FALLBACK = "source_read_fallback"
    TASK_SEARCH_COMPLETED = "direction_search_completed"
    TASK_EVIDENCE_ADDED = "direction_evidence_added"
    # 受理即发：任务已入队、可能还在等并发槽。界面据此先建一张"排队中"卡，
    # 与 RESEARCH_TASK_STARTED（真正开始跑）分开，避免排队任务对用户不可见。
    RESEARCH_TASK_QUEUED = "research_task_queued"
    RESEARCH_TASK_STARTED = "research_task_started"
    RESEARCH_TASK_COMPLETED = "research_task_completed"
    RESEARCH_TASK_FAILED = "research_task_failed"
    RESEARCH_ROUND_COMPLETED = "research_round_completed"
    DELEGATE_STARTED = "delegate_started"
    DELEGATE_COMPLETED = "delegate_completed"
    # 任务结果归入工作集的时刻；lost 变体兼作进程重启后待清算的记录。
    # 字面值保留历史写法：事件已入库，改值会让旧数据与新事件断开，只能改常量名。
    TASK_MERGED = "direction_arrival_merged"
    SUPERVISOR_MODEL_TURN = "supervisor_model_turn"
    CONTEXT_COMPACTED = "context_compacted"
    WRITER_EXHAUSTED = "writer_exhausted"
    WRITER_INLINE_DRAFT_RECOVERED = "writer_inline_draft_recovered"
    WRITER_INLINE_DRAFT_REJECTED = "writer_inline_draft_rejected"
    WRITER_DRAFT_VALIDATED = "writer_draft_validated"
    WRITER_DRAFT_READY = "writer_draft_ready"
    RUN_STATUS = "run_status"
    RUN_DONE = "run_done"
    RUN_HEADLINE_UPDATED = "run_headline_updated"
    CLARIFICATION_REQUESTED = "clarification_requested"
    TEXT_DELTA = "text_delta"
