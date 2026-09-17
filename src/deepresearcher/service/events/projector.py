"""事件 → 客户端安全帧 的纯函数投影。

**默认拒绝**：只处理映射表里显式列出的 event_type，未知类型返回 None——
引擎将来新增事件不会悄悄把 payload 漏到浏览器。每一帧都逐字段新建 dict，
绝不整包转发 payload；``error``/``*_preview``/``prompt`` 等敏感键在代码层面
就没有进入路径。

SSE 帧词汇表（与前端、服务层合成事件共用）。呈现原则：管线每个阶段是
一个阶段块（出现即开始、完成即灰化），阶段自己的判断文本进该块；方向卡是
Supervisor 块的子项。

- ``stage_open``  阶段开始 {stage, title}
- ``plan``        阶段旁白一行 {stage, text}（规划理由/决策/跳过原因）
- ``stage_done``  阶段结束+结论 {stage, status, text}
- ``tick``        全局杂项一行（取消、未知阶段）
- ``stats``       运行指标 → 标题区（轮次/方向/证据计数）
- ``task_open``   开一张方向卡 {task, title}
- ``task_update`` 方向卡滚动一条动作 {task, text}
- ``task_done``   方向卡收束 {task, status, summary}
- ``status``      运行状态变化（服务层合成）
- ``error``       需要用户知道的异常提示（仅安全文案）
- ``done``        终态 {status, answer_mode, report_available}

各 agent 的 model_turn 计数不面向用户；唯一例外是 supervisor 的规划文字
（写了字才出口，见 plan）。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from deepresearcher.routing import NodeName
from deepresearcher.schemas import StopReason

TRUNCATED_EVENT = "stream_truncated"

_STAGE_TITLES: dict[str, str] = {
    NodeName.ROUTER: "理解问题 · Router",
    NodeName.CLARIFY: "澄清范围 · Clarify",
    NodeName.QUICK_ANSWER: "即时回答",
    NodeName.SUPERVISOR: "研究规划 · Supervisor",
    NodeName.WRITER: "撰写报告 · Writer",
    NodeName.REFLECTION: "审阅报告 · Reviewer",
    NodeName.RENDER_FINAL_REPORT: "生成最终报告",
}

# 方向卡第二行动作：{event_type: 文案}。
_TASK_UPDATES: dict[str, str] = {
    "source_fetch_started": "读取来源中…",
    "source_fetch_completed": "来源读取完成",
    "direction_evidence_added": "证据入池 +{accepted_count}",
    "source_timeout": "来源读取超时",
    "source_read_failed": "某来源不可读，已跳过",
    "source_read_skipped": "来源重复，已跳过",
}


@dataclass(frozen=True)
class SseFrame:
    event: str
    data: dict


def project(record: Mapping[str, Any]) -> SseFrame | None:
    """引擎/合成事件 dict → 客户端帧；不在白名单内的一律 None。"""
    event_type = record.get("event_type")
    if not isinstance(event_type, str):
        return None
    payload = record.get("payload")
    if not isinstance(payload, Mapping):
        payload = {}
    seq = record.get("seq")
    node = record.get("node")

    # ---- 阶段生命周期 ----
    if event_type == "node_started":
        if not isinstance(node, str) or node not in _STAGE_TITLES:
            return _tick(seq, f"{node} 开始" if isinstance(node, str) else None)
        return _frame("stage_open", seq, {"stage": node, "title": _STAGE_TITLES[node]})
    if event_type == "node_completed":
        if isinstance(node, str) and node in _STAGE_TITLES:
            return _frame(
                "stage_done",
                seq,
                {"stage": node, "status": "done", "text": _stage_conclusion(node, payload)},
            )
        return None
    if event_type == "node_failed":
        # 只说哪个阶段失败：内部异常文本永远不出网关（完整信息在 RunEvent/日志）。
        # 已知节点必须用 failed 的 stage_done 关框——否则阶段框停在"运行中"
        # 的绿点上永远呼吸（reflection 内容审查事故实锤），未知节点退回全局错误行。
        if isinstance(node, str) and node in _STAGE_TITLES:
            return _frame(
                "stage_done",
                seq,
                {"stage": node, "status": "failed", "text": f"{_STAGE_TITLES[node]} 执行失败"},
            )
        return _frame(
            "error", seq, {"text": f"{node if isinstance(node, str) else '某阶段'} 阶段执行失败"}
        )
    if event_type == "node_cancelled":
        return _tick(seq, "该阶段已取消")

    # ---- Supervisor 决策旁白（plan 帧都带 stage 归属）----
    if event_type == "research_stopped":
        return _plan(
            seq, NodeName.SUPERVISOR, f"研究提前结束：{_stop_reason_text(payload.get('reason'))}"
        )
    if event_type == "delegate_completed":
        status = str(payload.get("status", ""))
        if status == "skipped":
            return _plan(seq, NodeName.SUPERVISOR, "发现重复研究方向，已跳过并调整计划")
        if status == "blocked":
            return _plan(seq, NodeName.SUPERVISOR, "研究轮次预算耗尽，开始收束")
        return None
    if event_type == "supervisor_model_turn":
        thought = _text(payload.get("content_preview"), 800)  # 与 _PREVIEW_CHARS 对齐
        return _plan(seq, NodeName.SUPERVISOR, thought) if thought else None
    # ---- 方向卡（Supervisor 块内子项）----
    if event_type == "research_task_started":
        title = _text(payload.get("question") or payload.get("research_direction"), 140)
        return _task("task_open", seq, payload, {"title": title})
    if event_type == "research_task_completed":
        summary = "证据 {} · 来源 {}".format(
            _int(payload.get("evidence_count")), _int(payload.get("source_count"))
        )
        return _task(
            "task_done",
            seq,
            payload,
            {"status": _text(payload.get("execution_status"), 16) or "done", "summary": summary},
        )
    if event_type == "research_task_failed":
        return _task("task_done", seq, payload, {"status": "failed", "summary": "研究未成功"})
    if event_type in _TASK_UPDATES:
        text = _TASK_UPDATES[event_type]
        if "{candidate_count}" in text:
            text = text.format(candidate_count=_int(payload.get("candidate_count")))
        if "{accepted_count}" in text:
            text = text.format(accepted_count=_int(payload.get("accepted_count")))
        return _task("task_update", seq, payload, {"text": text})
    if event_type == "direction_search_completed":
        text = "检索完成：{} 条候选来源".format(_int(payload.get("candidate_count")))
        return _task("task_update", seq, payload, {"text": text})

    # ---- 指标与杂项 ----
    if event_type == "research_round_completed":
        return _frame(
            "stats",
            seq,
            {
                "round": _int(payload.get("round")),
                "tasks_completed": _int(payload.get("completed_tasks")),
                "tasks_total": _int(payload.get("task_count")),
                "evidence_added": _int(payload.get("evidence_added")),
                "evidence_total": _int(payload.get("total_evidence_count")),
            },
        )
    if event_type == "text_delta":
        # 生产者是 RunExecutor 对官方 astream(subgraphs=True) messages 的 ns 路由；
        # 这里仍做第二道闸：只放行 supervisor。writer 正文走工具参数、
        # reflection 是结构化调用——两者只应看到最终聚合结果。
        channel = _text(payload.get("channel"), 24)
        text = _text(payload.get("text"), 200)
        if channel != "supervisor" or not text:
            return None
        return SseFrame(event="text_delta", data={"channel": channel, "text": text})
    if event_type == TRUNCATED_EVENT:
        return _frame(
            "error", seq, {"text": "实时推送拥塞，部分进度被跳过；刷新页面可回放完整进度"}
        )

    # ---- 服务层合成事件 ----
    if event_type == "run_status":
        status = _text(payload.get("status"), 32)
        return _frame("status", seq, {"status": status}) if status else None
    if event_type == "clarification_requested":
        question = _text(payload.get("question"), 500)
        raw_options = payload.get("options")
        options = (
            [_text(item, 120) for item in raw_options[:3] if _text(item, 120)]
            if isinstance(raw_options, list)
            else []
        )
        return (
            _frame("clarification", seq, {"question": question, "options": options})
            if question
            else None
        )
    if event_type == "run_done":
        return _frame(
            "done",
            seq,
            {
                "status": _text(payload.get("status"), 32),
                "answer_mode": _text(payload.get("answer_mode"), 32),
                "report_available": bool(payload.get("report_available")),
            },
        )
    return None


def _stage_conclusion(node: str, payload: Mapping[str, Any]) -> str:
    """阶段完成时的人话结论——只读取白名单键（各键均为有界标量）。"""
    if node == NodeName.ROUTER:
        route = str(payload.get("route", ""))
        label = {"deep_research": "进入深度研究", "quick_answer": "按即时回答处理"}.get(
            route, route
        )
        reason = _text(payload.get("route_reason"), 120)
        return "；".join(part for part in (label, reason) if part)
    if node == NodeName.CLARIFY:
        # Clarifier 的完成结论来自子图 finalize；没有触发 interrupt 时也应让
        # 用户看到 Agent 确认后的研究范围，而不是留下一个空阶段卡片。
        return _text(payload.get("research_brief"), 800) or _text(
            payload.get("clarified_query"), 300
        )
    if node == NodeName.SUPERVISOR:
        # 注意键名：节点产出的 evidences 列表增量在 summary 里是 evidences_count；
        # evidence_count 是 writer 才写的累计字段，supervisor 节点里恒为 0（曾致误报）。
        return "规划完成 · 第 {} 轮 · 本次新增证据 {}".format(
            _int(payload.get("current_round")), _int(payload.get("evidences_count"))
        )
    if node == NodeName.WRITER:
        return {
            "completed": "草稿通过校验",
            "exhausted": "写作未正常收束",
            "failed": "写作失败",
        }.get(str(payload.get("writer_status", "")), "")
    if node == NodeName.REFLECTION:
        label = {"approved": "审阅通过", "rejected": "审阅要求修改"}.get(
            str(payload.get("review_status", "")), ""
        )
        feedback = _text(payload.get("review_feedback"), 120)
        return "；".join(part for part in (label, feedback) if part)
    if node == NodeName.RENDER_FINAL_REPORT:
        chars = _int(payload.get("report_chars"))
        return f"报告 {chars} 字符" if chars else ""
    return ""


def _plan(seq: Any, stage: str, text: str) -> SseFrame | None:
    return _frame("plan", seq, {"stage": stage, "text": text}) if text else None


def _task(event: str, seq: Any, payload: Mapping[str, Any], extra: dict) -> SseFrame | None:
    task = _text(payload.get("task_id"), 64)
    if not task:
        return None
    return _frame(event, seq, {"task": task, **extra})


def _tick(seq: Any, text: str | None) -> SseFrame | None:
    return _frame("tick", seq, {"text": text}) if text else None


def _frame(event: str, seq: Any, data: dict) -> SseFrame:
    safe_seq = _int(seq) if isinstance(seq, int) or (isinstance(seq, str) and seq.isdigit()) else 0
    return SseFrame(event=event, data={**data, "seq": safe_seq})


def _text(value: Any, max_chars: int) -> str:
    return str(value)[:max_chars] if value is not None else ""


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _stop_reason_text(value: Any) -> str:
    if isinstance(value, str) and value:
        try:
            return StopReason(value).description
        except ValueError:
            return value[:60]
    return "未知原因"
