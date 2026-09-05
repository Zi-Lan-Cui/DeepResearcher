import json

import pytest

from deepsearch_agent.schemas import StopReason
from deepsearch_agent.service.events.projector import project

# 每个用例都会把 SECRET 塞进这些被禁字段，任何一帧的输出 JSON 里都不许出现。
DENIED_FIELDS = {
    "error": "SECRET-error",
    "content_preview": "SECRET-preview",
    "prompt": "SECRET-prompt",
    "markdown": "SECRET-markdown",
    "report": "SECRET-report",
    "queries": ["SECRET-query"],
    "review_issues": "SECRET-review",
    "coverage_gaps": "SECRET-gaps",
}


def _record(event_type, payload=None, **top):
    record = {"event_type": event_type, "run_id": "run-x", "seq": 7, **DENIED_FIELDS, **top}
    record["payload"] = {**(payload or {}), **DENIED_FIELDS}
    return record


def test_node_started_opens_stage_block():
    frame = project(_record("node_started", node="clarify"))
    assert frame is not None
    assert frame.event == "stage_open"
    assert frame.data["stage"] == "clarify"
    assert "澄清" in frame.data["title"]
    assert frame.data["seq"] == 7


def test_node_completed_closes_stage_with_conclusion():
    done = project(
        _record(
            "node_completed",
            {"route": "deep_research", "route_reason": "多主体对比问题"},
            node="router",
        )
    )
    assert (done.event, done.data["status"]) == ("stage_done", "done")
    assert done.data["text"].startswith("进入深度研究")
    clarified = project(
        _record(
            "node_completed",
            {
                "research_brief": "重点比较 Redis 在传统后端与 Agent 中的用途",
                "clarified_query": "Redis 有什么作用？",
            },
            node="clarify",
        )
    )
    assert clarified.data["text"] == "重点比较 Redis 在传统后端与 Agent 中的用途"
    clarify_fallback = project(
        _record(
            "node_completed",
            {"clarified_query": "Redis 有什么作用？"},
            node="clarify",
        )
    )
    assert clarify_fallback.data["text"] == "Redis 有什么作用？"
    writer = project(_record("node_completed", {"writer_status": "completed"}, node="writer"))
    assert writer.data["text"] == "草稿通过校验"
    # supervisor 读的是列表增量键 evidences_count（曾误读 evidence_count 恒为 0）
    sup = project(
        _record("node_completed", {"current_round": 2, "evidences_count": 49}, node="supervisor")
    )
    assert sup.data["text"] == "规划完成 · 第 2 轮 · 本次新增证据 49"
    reviewer = project(_record("node_completed", {"review_status": "approved"}, node="reflection"))
    assert reviewer.data["text"] == "审阅通过"
    # 未注册的收尾事件保持安静；SECRET 类字段无一透入
    assert project(_record("node_completed", node="unknown_node")) is None
    assert "SECRET" not in json.dumps(done.data, ensure_ascii=False)


def test_unknown_event_type_returns_none():
    assert project(_record("brand_new_engine_event", {"anything": 1})) is None


def test_unmapped_model_turn_agent_returns_none():
    assert project(_record("mystery_model_turn", {"turn": 1})) is None


def test_node_failed_uses_safe_text_only():
    frame = project(_record("node_failed", node="reflection"))
    # 已知节点：失败也必须关框（红点收束），绝不停留在 running
    assert (frame.event, frame.data["stage"], frame.data["status"]) == (
        "stage_done",
        "reflection",
        "failed",
    )
    assert "Reviewer" in frame.data["text"] and "执行失败" in frame.data["text"]
    assert "SECRET" not in json.dumps(frame.data, ensure_ascii=False)
    # 未知节点退回全局错误行
    assert project(_record("node_failed", node="mystery")).event == "error"


def test_direction_search_becomes_task_update_without_direction_text():
    """方向文案只在开卡时出口一次；update 行只报数量，防刷屏且不重复长标题。"""
    frame = project(
        _record(
            "direction_search_completed",
            {"task_id": "task-0001", "research_direction": "x" * 200, "candidate_count": 5},
        )
    )
    assert frame.event == "task_update"
    assert frame.data["task"] == "task-0001"
    assert frame.data["text"] == "检索完成：5 条候选来源"
    assert "xxxx" not in json.dumps(frame.data, ensure_ascii=False)


def test_task_frames_require_task_id():
    assert project(_record("direction_search_completed", {"candidate_count": 5})) is None


def test_research_round_completed_maps_to_stats_frame():
    """轮次统计进标题区指标条，不再混入结果流叙事。"""
    frame = project(
        _record(
            "research_round_completed",
            {
                "round": 2,
                "task_count": 4,
                "completed_tasks": 3,
                "evidence_added": 6,
                "total_evidence_count": 14,
            },
        )
    )
    assert frame.event == "stats"
    assert frame.data["round"] == 2
    assert (frame.data["tasks_completed"], frame.data["tasks_total"]) == (3, 4)
    assert (frame.data["evidence_added"], frame.data["evidence_total"]) == (6, 14)


def test_research_stopped_maps_stop_reason_description():
    frame = project(_record("research_stopped", {"reason": str(StopReason.ROUND_BUDGET_EXHAUSTED)}))
    assert frame.event == "plan" and frame.data["stage"] == "supervisor"
    assert StopReason.ROUND_BUDGET_EXHAUSTED.description in frame.data["text"]


def test_research_stopped_tolerates_unknown_reason():
    frame = project(_record("research_stopped", {"reason": "made-up"}))
    assert frame.data["text"].endswith("made-up")


@pytest.mark.parametrize("event_type", ["writer_model_turn", "researcher_model_turn"])
def test_non_supervisor_model_turn_events_are_silent(event_type):
    """轮次计数是内部预算视角；writer/researcher 的 turn 一律不出口。"""
    assert project(_record(event_type, {"turn": 3, "tool_names": ["SearchSources"]})) is None


def test_supervisor_turn_exports_only_written_thought():
    """Supervisor 写了规划文字才可见；只发 tool_call 的轮（preview 空）保持安静。

    不用 _record：DENIED_FIELDS 注入会覆盖 content_preview，这里要精确控制它。
    """
    base = {
        "event_type": "supervisor_model_turn",
        "run_id": "r",
        "seq": 7,
        "payload": {
            "turn": 2,
            "content_preview": "",
            "tool_names": ["ResearchDelegate"],
            "queries": ["SECRET-query"],
        },
    }
    assert project(base) is None
    base["payload"]["content_preview"] = "现有证据缺少法家视角，补派一个方向。"
    thought = project(base)
    assert thought.event == "plan"
    assert thought.data["text"] == "现有证据缺少法家视角，补派一个方向。"
    # 除文字外不携带其他 payload（tool_names/queries 仍在被禁面）
    assert set(thought.data) == {"stage", "text", "seq"}
    assert "SECRET" not in json.dumps(thought.data, ensure_ascii=False)


def test_node_started_supervisor_opens_stage():
    frame = project(_record("node_started", node="supervisor"))
    assert (frame.event, frame.data["stage"]) == ("stage_open", "supervisor")
    assert "Supervisor" in frame.data["title"]


def test_research_task_card_lifecycle():
    opened = project(
        _record(
            "research_task_started",
            {"task_id": "task-0001", "question": "性善论的论证结构" + "。" * 200},
        )
    )
    assert opened.event == "task_open"
    assert opened.data["task"] == "task-0001"
    assert len(opened.data["title"]) == 140  # 标题截断
    reading = project(_record("source_fetch_completed", {"task_id": "task-0001"}))
    assert (reading.event, reading.data["text"]) == ("task_update", "来源读取完成")
    extracting = project(
        _record("evidence_chunk_completed", {"task_id": "task-0001", "candidate_count": 4})
    )
    assert extracting.data["text"] == "证据抽取 +4"
    done = project(
        _record(
            "research_task_completed",
            {
                "task_id": "task-0001",
                "execution_status": "completed",
                "evidence_count": 7,
                "source_count": 3,
            },
        )
    )
    assert (done.event, done.data["status"], done.data["summary"]) == (
        "task_done",
        "completed",
        "证据 7 · 来源 3",
    )


def test_writer_stage_lifecycle():
    opened = project(_record("node_started", node="writer"))
    assert (opened.event, opened.data["stage"]) == ("stage_open", "writer")
    # 草稿 ready 是 Writer 内部事件；审阅阶段会自行开框，不在 Writer 中抢跑播报。
    assert project(_record("writer_draft_ready", {})) is None
    # writer 的内部 turn/finished 帧不再出口（收束交给 node_completed 的结论）
    assert project(_record("writer_agent_finished", {"stop_reason": "final_response"})) is None
    assert project(_record("writer_model_turn", {"turn": 2})) is None
    closed = project(_record("node_completed", {"writer_status": "exhausted"}, node="writer"))
    assert (closed.event, closed.data["text"]) == ("stage_done", "写作未正常收束")


def test_agent_finished_events_are_silent():
    """agent 内部收尾帧不进用户视图（阶段收束统一走 stage_done）。"""
    assert project(_record("supervisor_agent_finished", {"stop_reason": "final_response"})) is None
    assert (
        project(_record("researcher_agent_finished", {"stop_reason": "model_call_limit_exceeded"}))
        is None
    )


def test_delegate_completed_maps_silent_planner_outcomes():
    skipped = project(
        _record("delegate_completed", {"status": "skipped", "reason": "duplicate_or_budget"})
    )
    assert skipped.data["text"] == "发现重复研究方向，已跳过并调整计划"
    blocked = project(
        _record("delegate_completed", {"status": "blocked", "reason": "round_budget_exhausted"})
    )
    assert "预算耗尽" in blocked.data["text"]
    # 正常完成的研究委托由研究员自己的事件播报，delegate 帧保持安静
    assert (
        project(_record("delegate_completed", {"status": "completed", "evidence_count": 5})) is None
    )


def test_run_status_and_run_done_shape():
    status = project(_record("run_status", {"status": "running"}))
    assert status.event == "status" and status.data["status"] == "running"
    done = project(
        _record(
            "run_done",
            {"status": "completed", "answer_mode": "deep_research", "report_available": True},
        )
    )
    assert done.event == "done"
    assert set(done.data) == {"status", "answer_mode", "report_available", "seq"}


def test_clarification_projection_only_exposes_prompt_and_three_options():
    frame = project(
        _record(
            "clarification_requested",
            {
                "question": "你更关心什么？",
                "options": ["成本", "效果", "风险", "不应外泄"],
                "internal_reason": "secret",
            },
        )
    )
    assert frame.event == "clarification"
    assert frame.data["question"] == "你更关心什么？"
    assert frame.data["options"] == ["成本", "效果", "风险"]
    assert "internal_reason" not in frame.data


def test_text_delta_whitelist_and_no_seq():
    ok = project(_record("text_delta", {"channel": "supervisor", "text": "先梳理缺口，"}))
    assert (ok.event, ok.data) == ("text_delta", {"channel": "supervisor", "text": "先梳理缺口，"})
    # 只有 supervisor 的思考值得逐字预览；writer（正文是工具参数、散文字幕无价值）、
    # research_agent 与空文本一律不出口
    assert project(_record("text_delta", {"channel": "research_agent", "text": "x"})) is None
    assert project(_record("text_delta", {"channel": "writer", "text": "写报告中"})) is None
    assert project(_record("text_delta", {"channel": "supervisor", "text": ""})) is None
    assert "seq" not in ok.data and "SECRET" not in json.dumps(ok.data, ensure_ascii=False)


def test_truncation_marker_becomes_error_frame():
    frame = project(_record("stream_truncated"))
    assert frame.event == "error"
    assert "回放" in frame.data["text"]


@pytest.mark.parametrize(
    "event_type",
    [
        "node_started",
        "node_failed",
        "node_cancelled",
        "direction_search_completed",
        "research_round_completed",
        "research_stopped",
        "source_fetch_completed",
        "source_reader_completed",
        "evidence_chunk_completed",
        "writer_draft_ready",
        "run_status",
        "run_done",
        "stream_truncated",
    ],
)
def test_never_leaks_denied_fields(event_type):
    """对抗测试：所有被映射的事件里，被禁字段必须无一透出。"""
    frame = project(_record(event_type, {"node": "supervisor", "turn": 1, "status": "running"}))
    if frame is None:
        return
    assert "SECRET" not in json.dumps(frame.data, ensure_ascii=False)
