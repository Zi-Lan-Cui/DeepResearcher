import json

import pytest

from deepresearcher.service.events.projector import project

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
    reviewer = project(_record("node_completed", {"review_status": "approved"}, node="review"))
    assert reviewer.data["text"] == "审阅通过"
    # 未注册的收尾事件保持安静；SECRET 类字段无一透入
    assert project(_record("node_completed", node="unknown_node")) is None
    assert "SECRET" not in json.dumps(done.data, ensure_ascii=False)


def test_node_cancelled_closes_known_stage_block():
    frame = project(_record("node_cancelled", node="supervisor"))
    assert (frame.event, frame.data["stage"], frame.data["status"]) == (
        "stage_done",
        "supervisor",
        "cancelled",
    )
    assert frame.data["text"] == "该阶段已取消"
    # 未知节点没有阶段框可关，退回全局 tick。
    assert project(_record("node_cancelled", node="mystery")).event == "tick"


def test_run_headline_updated_projects_headline_frame():
    frame = project(_record("run_headline_updated", {"headline": "两方案成本效果对比"}))
    assert frame is not None
    assert (frame.event, frame.data["text"]) == ("headline", "两方案成本效果对比")
    assert project(_record("run_headline_updated", {"headline": ""})) is None


def test_unknown_event_type_returns_none():
    assert project(_record("brand_new_engine_event", {"anything": 1})) is None


def test_unmapped_model_turn_agent_returns_none():
    assert project(_record("mystery_model_turn", {"turn": 1})) is None


def test_node_failed_uses_safe_text_only():
    frame = project(_record("node_failed", node="review"))
    # 已知节点：失败也必须关框（红点收束），绝不停留在 running
    assert (frame.event, frame.data["stage"], frame.data["status"]) == (
        "stage_done",
        "review",
        "failed",
    )
    assert "Reviewer" in frame.data["text"] and "执行失败" in frame.data["text"]


def test_legacy_reflection_node_replays_as_review():
    # 枚举改值前的存量 DB 行 node="reflection":必须归一到当前审阅节点、投出同样的关框帧。
    frame = project(_record("node_completed", {"review_status": "approved"}, node="reflection"))
    assert frame.data["stage"] == "review"
    assert frame.data["text"] == "审阅通过"
    assert "SECRET" not in json.dumps(frame.data, ensure_ascii=False)
    # 未知节点退回全局错误行
    assert project(_record("node_failed", node="mystery")).event == "error"


def test_direction_search_becomes_task_update_without_direction_text():
    """方向文案只在事件卡片首次出现时输出一次；update 行只报数量，不重复长标题。"""
    frame = project(
        _record(
            "direction_search_completed",
            {"task_id": "task-0001", "research_direction": "x" * 200, "candidate_count": 5},
        )
    )
    assert frame.event == "task_update"
    assert frame.data["task"] == "task-0001"
    assert frame.data["text"] == "本批检索完成：5 条候选来源"
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
    assert len(opened.data["title"]) == 140  # 无短题的存量事件:回退截断契约
    titled = project(
        _record(
            "research_task_started",
            {
                "task_id": "task-0002",
                "title": "性善论的先验根据",
                "question": "从孟子四端出发论证性善论的先验根据,并与荀子性恶论划界" + "。" * 200,
            },
        )
    )
    assert titled.data["title"] == "性善论的先验根据"  # 短题优先,不再展示长契约
    started = project(
        _record(
            "source_fetch_started",
            {"task_id": "task-0001", "requested_url": "https://www.pku.edu.cn/x?q=SECRET"},
        )
    )
    assert started.data["text"] == "正在读取：pku.edu.cn"
    registered = project(
        _record(
            "source_document_registered",
            {
                "task_id": "task-0001",
                "requested_url": "https://a.example/p",
                "final_url": "https://b.example/p",
            },
        )
    )
    assert registered.data["text"] == "已读取：b.example"
    cached = project(
        _record(
            "source_document_registered",
            {
                "task_id": "task-0001",
                "requested_url": "https://iso25000.com/en/x",
                "final_url": "https://iso25000.com/en/x",
                "material_cache_hit": True,
            },
        )
    )
    assert cached.data["text"] == "复用来源：iso25000.com"  # 缓存命中不是重新抓取
    skipped = project(
        _record(
            "source_read_skipped",
            {
                "task_id": "task-0001",
                "url": "https://pay.example/x",
                "reason_code": "login_required",
            },
        )
    )
    assert skipped.data["text"] == "已跳过：pay.example（需登录）"
    unknown_skip = project(
        _record(
            "source_read_skipped",
            {"task_id": "task-0001", "url": "", "reason_code": "some_machine_code"},
        )
    )
    assert unknown_skip.data["text"] == "来源已跳过（不可读）"
    failed = project(
        _record("source_read_failed", {"task_id": "task-0001", "url": "https://dead.example"})
    )
    assert failed.data["text"] == "读取失败：dead.example"
    extracting = project(
        _record("direction_evidence_added", {"task_id": "task-0001", "accepted_count": 4})
    )
    assert extracting.data["text"] == "证据入池 +4"
    all_rejected = project(
        _record(
            "direction_evidence_added",
            {
                "task_id": "task-0001",
                "accepted_count": 0,
                "rejected_count": 3,
                "duplicate_count": 0,
            },
        )
    )
    assert all_rejected.data["text"] == "证据提交未通过，按回执修正中"
    empty_audit = project(_record("direction_evidence_added", {"task_id": "task-0001"}))
    assert empty_audit is None  # 零入池零退回的审计记录不打扰用户
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


# stream_truncated 的产生者(进程内直投)已删除;它留在下方 default-deny 名单里,
# 保证即便有代码再投这种帧,投影层也不出口。


def test_search_query_progress_becomes_task_update():
    started = project(_record("search_query_started", {"task_id": "t1", "query": "关键词甲"}))
    assert (started.event, started.data["text"]) == ("task_update", "正在检索：关键词甲")
    done = project(
        _record(
            "search_query_completed",
            {"task_id": "t1", "query": "关键词甲", "candidate_count": 12},
        )
    )
    assert done.data["text"] == "关键词甲 → 12 条候选"
    empty = project(
        _record(
            "search_query_completed", {"task_id": "t1", "query": "关键词甲", "candidate_count": 0}
        )
    )
    assert empty.data["text"] == "关键词甲 → 未见相关来源"  # 查净了没结果,不是失败
    failed = project(_record("search_query_failed", {"task_id": "t1", "query": "x"}))
    assert failed.data["text"].startswith("检索失败，已跳过")


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"candidate_count": 5, "failure_count": 0}, "本批检索完成：5 条候选来源"),
        (
            {"candidate_count": 3, "failure_count": 1},
            "本批检索完成：3 条候选来源，其中 1 个查询失败",
        ),
        ({"candidate_count": 0, "failure_count": 0}, "本批检索完成：未找到相关来源"),
        # 全部查询失败:逐查询行已逐条说明,批次行不再重复下结论。
        ({"candidate_count": 0, "failure_count": 2}, None),
        ({"candidate_count": 0, "failure_count": 2, "status": "failed"}, None),
        # 无逐查询细节可依托的批次失败,必须留一行,否则隐身。
        ({"candidate_count": 0, "failure_count": 0, "status": "failed"}, "检索失败"),
    ],
)
def test_direction_search_completed_distinguishes_empty_from_failed(payload, expected):
    """区分度:0 候选不再一律"完成";;批次行只说细节行说不出来的事。"""
    frame = project(_record("direction_search_completed", {"task_id": "t1", **payload}))
    if expected is None:
        assert frame is None
    else:
        assert frame.data["text"] == expected
    # 存量事件(无 failure_count/status)仍走旧口径,不因新分支变化。
    legacy = project(_record("direction_search_completed", {"task_id": "t1", "candidate_count": 5}))
    assert legacy.data["text"] == "本批检索完成：5 条候选来源"


@pytest.mark.parametrize(
    "event_type",
    [
        "node_started",
        "node_failed",
        "node_cancelled",
        "direction_search_completed",
        "research_round_completed",
        "research_stopped",  # 无生产者的历史事件名:必须仍默认拒绝
        "source_fetch_started",
        "source_document_registered",
        "source_read_failed",
        "source_read_skipped",
        "direction_evidence_added",
        "writer_draft_ready",
        "run_status",
        "run_done",
        "stream_truncated",
        "search_query_started",
        "search_query_completed",
        "search_query_failed",
    ],
)
def test_never_leaks_denied_fields(event_type):
    """对抗测试：所有被映射的事件里，被禁字段必须无一透出。"""
    frame = project(_record(event_type, {"node": "supervisor", "turn": 1, "status": "running"}))
    if frame is None:
        return
    assert "SECRET" not in json.dumps(frame.data, ensure_ascii=False)
