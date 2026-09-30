"""账本事件名的登记表(跨层 wire 契约)。

发射侧(instrumentation、agents/tools 的 emit 调用、hub 服务层合成)与
消费侧(projector 投影白名单、SSE 尾随、DB 查询)都引用这里的符号,改名
不再靠两侧字面量对撞。动态拼接的家族名以约定记录、不进枚举成员:
节点生命周期 `node_{status}`(observability/events/models.make_node_event)、
工具边界 `tool_{status}`(make_tool_event)、各 Agent 回合与工具事件
`{slug}_model_turn` / `{slug}_tool_{status}`(agents/middleware/observability)。
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
    DIRECTION_SEARCH_COMPLETED = "direction_search_completed"
    DIRECTION_EVIDENCE_ADDED = "direction_evidence_added"
    RESEARCH_TASK_STARTED = "research_task_started"
    RESEARCH_TASK_COMPLETED = "research_task_completed"
    RESEARCH_TASK_FAILED = "research_task_failed"
    RESEARCH_ROUND_COMPLETED = "research_round_completed"
    DELEGATE_STARTED = "delegate_started"
    DELEGATE_COMPLETED = "delegate_completed"
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
