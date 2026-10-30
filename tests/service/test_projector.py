import json

import pytest

from deepresearcher.service.web.presenters import report_kind
from deepresearcher.service.web.projector import project, project_review, project_run_error

# 每个用例都会把 SECRET 塞进这些被禁字段，任何一帧的输出 JSON 里都不许出现。
DENIED_FIELDS = {
    "error": "SECRET-error",
    "content_preview": "SECRET-preview",
    "prompt": "SECRET-prompt",
    "markdown": "SECRET-markdown",
    "report": "SECRET-report",
    "queries": ["SECRET-query"],
    "review_issues": "SECRET-review",
    "failure_details": ["SECRET-failures"],
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
                "intent_summary": "重点比较 Redis 在传统后端与 Agent 中的用途",
                "clarified_query": "Redis 有什么作用？",
            },
            node="clarify",
        )
    )
    assert clarified.data["text"] == "重点比较 Redis 在传统后端与 Agent 中的用途"
    # 澄清结论行有上界：再长的意图摘要也只投一行截断文本，不倾倒文书。
    long_intent = project(_record("node_completed", {"intent_summary": "长" * 200}, node="clarify"))
    assert len(long_intent.data["text"]) == 160
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
    # 读取的是增量键 evidences_count；误读累计键 evidence_count 会让数字恒为 0
    sup = project(
        _record("node_completed", {"current_round": 2, "evidences_count": 49}, node="supervisor")
    )
    assert sup.data["text"] == "规划完成 · 第 2 轮 · 本次新增证据 49"
    reviewer = project(_record("node_completed", {"review_status": "approved"}, node="review"))
    assert reviewer.data["text"] == "审阅通过"
    # 未注册的收尾事件不出口；SECRET 类字段均不透出
    assert project(_record("node_completed", node="unknown_node")) is None
    assert "SECRET" not in json.dumps(done.data, ensure_ascii=False)


def test_review_stage_delivers_its_own_summary_not_the_feedback():
    """卡片读 summary，不读 feedback。"""
    frame = project(
        _record(
            "node_completed",
            {
                "review_status": "approved",
                "review_summary": "核心结论均有来源支撑，仅负载曲线一处为推理。",
                "review_feedback": "报告围绕问题给出了完整回答：按目的选型……",
            },
            node="review",
        )
    )
    assert frame is not None
    assert frame.data["status"] == "done"
    assert frame.data["text"] == "审阅通过；核心结论均有来源支撑，仅负载曲线一处为推理。"


def test_review_rejection_is_warn_not_done():
    """退回修改走 warn，不与 failed 同档。"""
    frame = project(
        _record(
            "node_completed",
            {"review_status": "rejected", "review_summary": "第 3 章的 SLI 定义缺少权威出处。"},
            node="review",
        )
    )
    assert frame is not None
    assert frame.data["status"] == "warn"
    assert frame.data["text"].startswith("审阅要求修改")


def test_review_stage_falls_back_to_feedback_without_summary():
    """老检查点没有 summary 时不能变成空卡。"""
    frame = project(
        _record(
            "node_completed",
            {"review_status": "approved", "review_feedback": "报告可交付。"},
            node="review",
        )
    )
    assert frame.data["text"] == "审阅通过；报告可交付。"


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
    # 已知节点失败也要关框，状态标 failed，不能停留在 running
    assert (frame.event, frame.data["stage"], frame.data["status"]) == (
        "stage_done",
        "review",
        "failed",
    )
    assert "Reviewer" in frame.data["text"] and "执行失败" in frame.data["text"]


def test_unknown_node_failure_falls_back_to_global_error():
    # 未知节点没有阶段标题，只能退回全局错误行，不能停在「运行中」。
    assert project(_record("node_failed", node="mystery")).event == "error"


def test_batch_search_row_is_dropped_when_every_query_has_its_own_row():
    """逐查询行已给出候选数与失败数，批次汇总再报一遍只是重复。"""
    assert (
        project(
            _record("direction_search_completed", {"task_id": "task-0001", "candidate_count": 5})
        )
        is None
    )
    # 方向长标题不进 update 行，卡片标题只在建卡时给一次。
    assert (
        project(
            _record(
                "direction_search_completed",
                {"task_id": "task-0001", "research_direction": "x" * 200, "cache_hit_count": 1},
            )
        )
        is not None
    )


def test_batch_search_reports_cache_hits_without_query_rows():
    """命中进程内缓存的查询不发逐查询事件，只有批次行能告知这部分结果。"""
    frame = project(
        _record(
            "direction_search_completed",
            {"task_id": "task-0001", "candidate_count": 5, "cache_hit_count": 2},
        )
    )
    assert frame is not None and frame.event == "task_update"
    assert frame.data["step"] == {
        "act": "复用",
        "obj": "2 个查询",
        "res": "5 条候选",
        "note": "命中缓存",
    }
    assert "xxxx" not in json.dumps(frame.data, ensure_ascii=False)


def test_task_frames_require_task_id():
    assert (
        project(_record("direction_search_completed", {"candidate_count": 5, "cache_hit_count": 1}))
        is None
    )


def test_direction_arrival_merged_maps_to_plan_line():
    frame = project(
        _record(
            "direction_arrival_merged",
            {
                "task_id": "task-0002",
                "display_title": "复利数学原理",
                "execution_status": "completed",
                "evidence_count": 5,
                "running_task_ids": ["task-0003", "task-0004"],
                "merged_via": "model_turn",
            },
        )
    )
    assert frame is not None and frame.event == "plan"
    assert frame.data["stage"] == "supervisor"
    assert frame.data["text"] == ("任务「复利数学原理」已回到工作台（证据 5），还有 2 个任务在途")


def test_direction_arrival_lost_variant_maps_to_plan_line():
    frame = project(
        _record(
            "direction_arrival_merged",
            {
                "task_id": "task-0002",
                "display_title": "复利数学原理",
                "execution_status": "failed",
                "evidence_count": 0,
                "running_task_ids": [],
                "merged_via": "lost",
            },
        )
    )
    assert frame is not None and frame.event == "plan"
    assert "进程重启丢失" in frame.data["text"]


def test_research_round_completed_maps_to_stats_frame():
    """轮次统计投影为指标条帧,不进结果流叙事。"""
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
    """轮次计数是内部预算信息,writer/researcher 的 turn 不出口。"""
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
    # 除文字外不携带其他 payload,tool_names 与 queries 仍在被禁清单
    assert set(thought.data) == {"stage", "text", "seq"}
    assert "SECRET" not in json.dumps(thought.data, ensure_ascii=False)


def test_supervisor_turn_dumping_tool_markup_is_dropped():
    """网关把工具调用体当正文吐出时，整条旁白不出帧——协议文本不进用户时间线。"""
    leaked = (
        "我先基于用户原话拆解缺口。"
        + "<function=ResearchDelegate><parameter=display_title>"
        + "复利话术的骗局包装方式</function> null"
    )
    assert (
        project(
            {
                "event_type": "supervisor_model_turn",
                "run_id": "r",
                "seq": 8,
                "payload": {"turn": 3, "content_preview": leaked, "tool_names": []},
            }
        )
        is None
    )
    assert (
        project(
            {
                "event_type": "supervisor_model_turn",
                "run_id": "r",
                "seq": 9,
                "payload": {"turn": 3, "content_preview": "<antml:parameter>x", "tool_names": []},
            }
        )
        is None
    )
    normal = project(
        {
            "event_type": "supervisor_model_turn",
            "run_id": "r",
            "seq": 10,
            "payload": {
                "turn": 3,
                "content_preview": "先读《经济学人》的两篇综述。",
                "tool_names": [],
            },
        }
    )
    assert normal is not None and normal.event == "plan"


def test_node_started_supervisor_opens_stage():
    frame = project(_record("node_started", node="supervisor"))
    assert (frame.event, frame.data["stage"]) == ("stage_open", "supervisor")
    assert "Supervisor" in frame.data["title"]


def test_queued_task_opens_card_before_concurrency_slot():
    """受理即建卡：排队帧与开始帧同形，只多一个 state，前端据此显示"排队中"。"""
    queued = project(
        _record(
            "research_task_queued",
            {"task_id": "task-0003", "title": "持久化取舍"},
        )
    )
    assert (queued.event, queued.data["task"], queued.data["state"]) == (
        "task_open",
        "task-0003",
        "queued",
    )
    started = project(
        _record("research_task_started", {"task_id": "task-0003", "title": "持久化取舍"})
    )
    assert started.event == "task_open"
    assert "state" not in started.data  # 开始帧不带 state，缺省即进行中


def test_long_source_path_keeps_its_tail():
    """同一站点的页面靠路径尾部区分，截断保留尾部并落在路径分隔处。"""
    frame = project(
        _record(
            "source_document_registered",
            {
                "task_id": "task-0001",
                "requested_url": (
                    "https://redis.io/docs/latest/operate/oss_and_stack/management/persistence/"
                ),
            },
        )
    )
    shown = frame.data["step"]["obj"]
    assert len(shown) <= 48
    assert shown.startswith("redis.io/…/")
    assert shown.endswith("management/persistence")


def test_research_task_card_lifecycle():
    opened = project(
        _record(
            "research_task_started",
            {"task_id": "task-0001", "question": "性善论的论证结构" + "。" * 200},
        )
    )
    assert opened.event == "task_open"
    assert opened.data["task"] == "task-0001"
    assert len(opened.data["title"]) == 140  # 无短题时回退截断到 140 字，存量事件的契约
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
    assert titled.data["title"] == "性善论的先验根据"  # 有短题优先用短题，不展示长问题
    # 抓取开始不单独成行：它和随后的登记是同一件事，只留登记那一行。
    assert (
        project(
            _record(
                "source_fetch_started",
                {"task_id": "task-0001", "requested_url": "https://www.pku.edu.cn/x?q=SECRET"},
            )
        )
        is None
    )
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
    assert registered.data["step"] == {
        "act": "读取",
        "obj": "b.example/p",
        "res": "",
        "title": "https://b.example/p",
    }
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
    # 缓存命中不是重新抓取，动作词必须区分开。
    assert cached.data["step"]["act"] == "复用"
    assert cached.data["step"]["obj"] == "iso25000.com/en/x"
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
    # 原因是非计数信息，进括注，右列只留计数。
    assert skipped.data["step"] == {
        "act": "跳过",
        "obj": "pay.example",
        "res": "",
        "note": "需登录",
        "title": "https://pay.example/x",
        "tone": "dim",
    }
    unknown_skip = project(
        _record(
            "source_read_skipped",
            {"task_id": "task-0001", "url": "", "reason_code": "some_machine_code"},
        )
    )
    assert unknown_skip.data["step"] == {
        "act": "跳过",
        "obj": "一份来源",
        "res": "",
        "note": "不可读",
        "tone": "dim",
    }
    failed = project(
        _record("source_read_failed", {"task_id": "task-0001", "url": "https://dead.example"})
    )
    # 左列仍是动作词（呈现层拿它拼「正在…」），异常与否由 tone 标记。
    assert failed.data["step"]["act"] == "读取"
    assert failed.data["step"]["tone"] == "bad"
    assert failed.data["step"]["note"] == "读取失败"
    extracting = project(
        _record("direction_evidence_added", {"task_id": "task-0001", "accepted_count": 4})
    )
    assert extracting.data["step"] == {"act": "提交", "obj": "证据", "res": "4 条通过"}
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
    assert all_rejected.data["step"] == {
        "act": "提交",
        "obj": "证据",
        "res": "",
        "note": "未通过校验",
    }
    empty_audit = project(_record("direction_evidence_added", {"task_id": "task-0001"}))
    assert empty_audit is None  # 零新增零退回的审计记录不打扰用户
    done = project(
        _record(
            "research_task_completed",
            {
                "task_id": "task-0001",
                "execution_status": "completed",
                "evidence_count": 7,
                "source_count": 3,
                "finding": "Redis 复制默认异步，主从切换可能丢写。",
                "conclusion": "两条官方文档互证 Redis 的复制是异步的。",
            },
        )
    )
    assert (done.event, done.data["status"], done.data["summary"]) == (
        "task_done",
        "completed",
        "证据 7 · 来源 3",
    )
    # 摘要与结论是两条通道：前者上卡面，后者留在展开区。
    assert done.data["finding"] == "Redis 复制默认异步，主从切换可能丢写。"
    assert done.data["conclusion"] == "两条官方文档互证 Redis 的复制是异步的。"
    failed = project(
        _record("research_task_completed", {"task_id": "task-0001", "execution_status": "failed"})
    )
    assert failed.data["conclusion"] == ""  # 无结论的任务不编占位文本
    assert failed.data["finding"] == ""


def test_writer_stage_lifecycle():
    opened = project(_record("node_started", node="writer"))
    assert (opened.event, opened.data["stage"]) == ("stage_open", "writer")
    # 草稿 ready 是 Writer 内部事件；审阅阶段自行开框，不在 Writer 阶段提前播报。
    assert project(_record("writer_draft_ready", {})) is None
    # writer 内部 turn/finished 帧不对外输出；阶段结束由 node_completed 承载
    assert project(_record("writer_agent_finished", {"stop_reason": "final_response"})) is None
    assert project(_record("writer_model_turn", {"turn": 2})) is None
    closed = project(_record("node_completed", {"writer_status": "exhausted"}, node="writer"))
    assert (closed.event, closed.data["text"]) == ("stage_done", "写作轮次耗尽")


def test_agent_finished_events_are_silent():
    """agent 内部收尾帧不进用户视图，阶段结束统一走 stage_done。"""
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
    # 逐字预览只开给 supervisor;writer 正文是工具参数，
    # research_agent 与空文本都不出口
    assert project(_record("text_delta", {"channel": "research_agent", "text": "x"})) is None
    assert project(_record("text_delta", {"channel": "writer", "text": "写报告中"})) is None
    assert project(_record("text_delta", {"channel": "supervisor", "text": ""})) is None
    assert "seq" not in ok.data and "SECRET" not in json.dumps(ok.data, ensure_ascii=False)


# stream_truncated 已无生产者，仍留在下方 default-deny 名单里：
# 若有代码重新投影这类帧，出口仍会被拒绝。


def test_search_query_progress_becomes_single_step_row():
    """检索开始不单独成行：它和同一查询的结果是同一件事。"""
    assert project(_record("search_query_started", {"task_id": "t1", "query": "关键词甲"})) is None
    done = project(
        _record(
            "search_query_completed",
            {"task_id": "t1", "query": "关键词甲", "candidate_count": 12},
        )
    )
    assert done.data["step"] == {"act": "检索", "obj": "「关键词甲」", "res": "12 条候选"}
    empty = project(
        _record(
            "search_query_completed", {"task_id": "t1", "query": "关键词甲", "candidate_count": 0}
        )
    )
    # 零候选是检索无结果，不是失败：右列留空，原因进括注。
    assert empty.data["step"] == {
        "act": "检索",
        "obj": "「关键词甲」",
        "res": "",
        "note": "未见相关来源",
    }
    failed = project(_record("search_query_failed", {"task_id": "t1", "query": "x"}))
    # 与读取失败同档：动作词不动，只把状态词染红。
    assert failed.data["step"] == {
        "act": "检索",
        "obj": "「x」",
        "res": "",
        "note": "失败",
        "tone": "bad",
    }


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        # 没有逐查询行可依托的整批失败必须留一行，否则用户看不到失败。
        ({"status": "failed"}, "失败"),
        ({"status": "failed", "failure_count": 2}, "失败"),
        # 逐查询行齐全且无缓存命中：批次行不再重复汇总。
        ({"candidate_count": 5}, None),
        ({"candidate_count": 0}, None),
        ({"candidate_count": 3, "failure_count": 1}, None),
        # 命中缓存的查询没有逐查询行，批次行是它唯一的出口。
        ({"candidate_count": 5, "cache_hit_count": 1}, "命中缓存"),
    ],
)
def test_batch_search_only_reports_what_query_rows_cannot(payload, expected):
    """0 候选不报完成；逐查询行覆盖得到的汇总一律省掉。"""
    frame = project(_record("direction_search_completed", {"task_id": "t1", **payload}))
    if expected is None:
        assert frame is None
    else:
        assert frame.data["step"]["note"] == expected


def test_search_skipped_row_tells_quota_apart_from_duplicates():
    """被拦的检索只能靠这一行留痕。"""
    quota = project(
        _record(
            "direction_search_skipped",
            {
                "task_id": "t1",
                "reason": "query_quota_exhausted",
                "queries_used": 6,
                "query_limit": 6,
            },
        )
    )
    assert quota is not None
    assert quota.event == "task_update"
    assert quota.data["step"]["act"] == "检索"
    assert quota.data["step"]["note"] == "额度已用尽 6/6"
    assert quota.data["step"]["tone"] == "bad"

    duplicated = project(
        _record("direction_search_skipped", {"task_id": "t1", "reason": "no_novel_queries"})
    )
    assert duplicated is not None
    assert duplicated.data["step"]["note"] == "与已执行查询全部重复"


@pytest.mark.parametrize(
    "event_type",
    [
        "node_started",
        "node_failed",
        "node_cancelled",
        "direction_search_completed",
        "research_round_completed",
        "research_stopped",  # 无生产者的历史事件名：必须仍默认拒绝
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
        "direction_arrival_merged",
    ],
)
def test_never_leaks_denied_fields(event_type):
    """所有会被投影的事件,被禁字段一律不透出。"""
    frame = project(_record(event_type, {"node": "supervisor", "turn": 1, "status": "running"}))
    if frame is None:
        return
    assert "SECRET" not in json.dumps(frame.data, ensure_ascii=False)


def test_report_kind_separates_shell_from_real_body():
    """正文形态决定徽章与卡片标题，空壳页不能因为换了词就变成一份报告。"""
    assert report_kind(None) == "none"
    assert report_kind("") == "none"
    assert report_kind("# 报告未生成\n\n## 研究问题\n…") == "none"
    assert report_kind("# 研究报告\n\n正文") == "full"


def test_project_run_error_translates_stage_and_drops_internals():
    """阶段名要与时间线同一个词；detail 是异常原文，不外发。"""
    error = project_run_error(
        {
            "stage": "writer",
            "code": "node_failed",
            "message": "与模型服务的连接被重置",
            "retryable": True,
            "detail": "SECRET",
        }
    )
    assert error == {
        "stage": "writer",
        "stage_label": "撰写报告 · Writer",
        "code": "node_failed",
        "message": "与模型服务的连接被重置",
    }


def test_project_run_error_falls_back_to_raw_stage_and_skips_empty():
    assert (
        project_run_error({"stage": "custom", "code": "run_exception", "message": "失败"})[
            "stage_label"
        ]
        == "custom"
    )
    assert project_run_error(None) is None
    assert project_run_error({"stage": "writer"}) is None


def test_project_review_lists_opinions_and_skips_pending():
    review = project_review(
        {
            "status": "rejected",
            "attempts": 2,
            "feedback": "整体未通过",
            "failure_reason": "",
            "issues": [{"severity": "fatal", "reason": f"意见{i}"} for i in range(5)],
        }
    )
    assert review is not None
    assert review["attempts"] == 2
    assert review["issues"] == ["意见0", "意见1", "意见2"]  # 最多三条，正文列不下更多
    assert project_review({"status": "pending", "attempts": 0}) is None
    assert project_review(None) is None
