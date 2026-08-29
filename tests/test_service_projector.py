import json

import pytest

from deepsearch_agent.schemas import StopReason
from deepsearch_agent.service.projector import project

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


def test_node_started_maps_to_chinese_narration():
    frame = project(_record("node_started", node="supervisor"))
    assert frame is not None
    assert frame.event == "tick"
    assert frame.data["text"] == "正在拆解研究任务…"
    assert frame.data["seq"] == 7


def test_unknown_event_type_returns_none():
    assert project(_record("brand_new_engine_event", {"anything": 1})) is None


def test_unmapped_model_turn_agent_returns_none():
    assert project(_record("mystery_model_turn", {"turn": 1})) is None


def test_node_failed_uses_safe_text_only():
    frame = project(_record("node_failed", node="writer"))
    assert frame.event == "error"
    assert frame.data["text"] == "writer 阶段执行失败"
    assert "SECRET" not in json.dumps(frame.data, ensure_ascii=False)


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
    frame = project(
        _record("research_stopped", {"reason": str(StopReason.ROUND_BUDGET_EXHAUSTED)})
    )
    assert StopReason.ROUND_BUDGET_EXHAUSTED.description in frame.data["text"]


def test_research_stopped_tolerates_unknown_reason():
    frame = project(_record("research_stopped", {"reason": "made-up"}))
    assert frame.data["text"].endswith("made-up")


@pytest.mark.parametrize(
    "event_type", ["supervisor_model_turn", "writer_model_turn", "researcher_model_turn"]
)
def test_model_turn_events_are_silent(event_type):
    """轮次计数是内部预算视角，不再面向用户（第 N 轮刷屏曾被指不友好）。"""
    assert project(_record(event_type, {"turn": 3, "tool_names": ["SearchSources"]})) is None


def test_research_task_card_lifecycle():
    opened = project(
        _record("research_task_started", {"task_id": "task-0001", "question": "性善论的论证结构" + "。" * 200})
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


def test_writer_collapses_to_single_card():
    opened = project(_record("node_started", node="writer"))
    assert (opened.event, opened.data["task"], opened.data["title"]) == (
        "task_open",
        "_writer",
        "撰写报告",
    )
    ready = project(_record("writer_draft_ready", {}))
    assert (ready.event, ready.data["text"]) == ("task_update", "草稿完成，进入审阅")
    finished = project(_record("writer_agent_finished", {"stop_reason": "final_response"}))
    assert (finished.event, finished.data["status"]) == ("task_done", "done")
    exhausted = project(_record("writer_agent_finished", {"stop_reason": "model_call_limit_exceeded"}))
    assert exhausted.data["status"] == "warn"


def test_agent_finished_events_are_silent():
    """supervisor/researcher 的收尾帧不进用户视图（writer 的走卡片收束）。"""
    assert project(_record("supervisor_agent_finished", {"stop_reason": "final_response"})) is None
    assert project(_record("researcher_agent_finished", {"stop_reason": "model_call_limit_exceeded"})) is None


def test_delegate_completed_maps_silent_planner_outcomes():
    skipped = project(_record("delegate_completed", {"status": "skipped", "reason": "duplicate_or_budget"}))
    assert skipped.data["text"] == "发现重复研究方向，已跳过并调整计划"
    blocked = project(_record("delegate_completed", {"status": "blocked", "reason": "round_budget_exhausted"}))
    assert "预算耗尽" in blocked.data["text"]
    # 正常完成的研究委托由研究员自己的事件播报，delegate 帧保持安静
    assert project(_record("delegate_completed", {"status": "completed", "evidence_count": 5})) is None


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
        "supervisor_model_turn",
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
