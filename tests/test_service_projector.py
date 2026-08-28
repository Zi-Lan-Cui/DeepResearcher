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


def test_direction_search_completed_shows_direction_and_count():
    long_direction = "性" * 100
    frame = project(
        _record(
            "direction_search_completed",
            {"research_direction": long_direction, "candidate_count": 5, "queries": ["x"]},
        )
    )
    assert frame.data["text"].count("性") == 60  # 截到 60 字
    assert "5 条候选来源" in frame.data["text"]
    assert "SECRET" not in json.dumps(frame.data, ensure_ascii=False)


def test_research_round_completed_counts():
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
    assert frame.data["text"] == "第 2 轮研究完成：方向 3/4，新增证据 6（累计 14）"


def test_research_stopped_maps_stop_reason_description():
    frame = project(
        _record("research_stopped", {"reason": str(StopReason.ROUND_BUDGET_EXHAUSTED)})
    )
    assert StopReason.ROUND_BUDGET_EXHAUSTED.description in frame.data["text"]


def test_research_stopped_tolerates_unknown_reason():
    frame = project(_record("research_stopped", {"reason": "made-up"}))
    assert frame.data["text"].endswith("made-up")


@pytest.mark.parametrize(
    ("event_type", "label"),
    [("supervisor_model_turn", "研究规划"), ("writer_model_turn", "报告撰写"),
     ("researcher_model_turn", "方向检索")],
)
def test_model_turn_generic_line(event_type, label):
    frame = project(_record(event_type, {"turn": 3, "tool_names": ["SearchSources"]}))
    assert frame.data["text"] == f"{label}中（第 3 轮）"
    assert "SearchSources" not in json.dumps(frame.data, ensure_ascii=False)


def test_agent_finished_events_are_silent():
    assert project(_record("writer_agent_finished", {"stop_reason": "final_response"})) is None


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
