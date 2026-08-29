"""事件 → 客户端安全帧 的纯函数投影。

**默认拒绝**：只处理映射表里显式列出的 event_type，未知类型返回 None——
引擎将来新增事件不会悄悄把 payload 漏到浏览器。每一帧都逐字段新建 dict，
绝不整包转发 payload；``error``/``*_preview``/``prompt``/``queries`` 全列表等
敏感键在代码层面就没有进入路径。

SSE 帧词汇表（与前端、RunManager 合成事件共用）。呈现原则：用户看整体，
不被细节淹没——

- ``tick``     全局时间线一行（规划器/阶段级叙述）
- ``stats``    运行指标（轮次/方向/证据计数）→ 前端渲染到标题区，不进结果流
- ``task_open``  开一张方向卡 {task, title}
- ``task_update`` 卡片第二行滚动一条动作 {task, text}
- ``task_done``   卡片收束 {task, status, summary}
- ``status``   运行状态变化（RunManager 合成）
- ``error``    需要用户知道的异常提示（仅安全文案）
- ``done``     终态，data 带 {status, answer_mode, report_available}

各 agent 的 model_turn 计数**不再面向用户**（第 N 轮是内部预算视角），
writer 收敛为一张普通卡。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from deepsearch_agent.routing import NodeName
from deepsearch_agent.schemas import StopReason

TRUNCATED_EVENT = "stream_truncated"

# 无真实 task_id 的阶段也用卡片通道呈现（前端统一一套渲染）。
_WRITER_CARD_ID = "_writer"

_NODE_NARRATION: dict[str, str] = {
    NodeName.ROUTER: "正在理解问题…",
    NodeName.CLARIFY: "正在澄清研究范围…",
    NodeName.QUICK_ANSWER: "正在准备即时回答…",
    NodeName.SUPERVISOR: "正在拆解研究任务…",
    NodeName.REFLECTION: "正在审阅报告…",
    NodeName.RENDER_FINAL_REPORT: "正在生成最终报告…",
}

# 卡片第二行动作文案：{event_type: 模板函数}
_TASK_UPDATES: dict[str, str] = {
    "source_fetch_started": "读取来源中…",
    "source_fetch_completed": "来源读取完成",
    "source_reader_completed": "来源读取完成",
    "evidence_chunk_completed": "证据抽取 +{candidate_count}",
    "evidence_extraction_failed": "部分来源抽取失败",
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

    # ---- 方向卡生命周期 ----
    if event_type == "research_task_started":
        title = _text(payload.get("question") or payload.get("research_direction"), 140)
        return _task("task_open", seq, payload, title)
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
        return _task("task_update", seq, payload, {"text": text})
    if event_type == "direction_search_completed":
        text = "检索完成：{} 条候选来源".format(_int(payload.get("candidate_count")))
        return _task("task_update", seq, payload, {"text": text})

    # ---- Writer：一张普通卡，轮次噪声不出口 ----
    if event_type == "node_started" and record.get("node") == NodeName.WRITER:
        frame = _frame(
            "task_open", seq, {"task": _WRITER_CARD_ID, "title": "撰写报告"}
        )
        return frame
    if event_type == "writer_draft_ready":
        return _frame(
            "task_update", seq, {"task": _WRITER_CARD_ID, "text": "草稿完成，进入审阅"}
        )
    if event_type == "writer_agent_finished":
        stop = str(payload.get("stop_reason") or "")
        status = "done" if stop == "final_response" else "warn"
        return _frame(
            "task_done",
            seq,
            {"task": _WRITER_CARD_ID, "status": status, "summary": ""},
        )

    # ---- 全局时间线（整体视角）----
    if event_type == "node_started":
        node = record.get("node")
        text = _NODE_NARRATION.get(node, f"{node} 开始") if isinstance(node, str) else None
        return _tick(seq, text)
    if event_type == "node_failed":
        node = record.get("node")
        # 只说哪个阶段失败：内部异常文本永远不出网关（完整信息在 RunEvent/日志）。
        return _frame("error", seq, {"text": f"{node} 阶段执行失败" if isinstance(node, str) else "某阶段执行失败"})
    if event_type == "node_cancelled":
        return _tick(seq, "该阶段已取消")
    if event_type == "research_round_completed":
        # 轮次统计是运行指标不是进度叙事——渲染到标题区指标条，不混进结果流。
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
    if event_type == "research_stopped":
        return _tick(seq, f"研究提前结束：{_stop_reason_text(payload.get('reason'))}")
    if event_type == "delegate_completed":
        # 规划器被静默消化的工具调用（重复方向/预算闸口）——让“空轮次”在直播里可见。
        status = str(payload.get("status", ""))
        if status == "skipped":
            return _tick(seq, "发现重复研究方向，已跳过并调整计划")
        if status == "blocked":
            return _tick(seq, "研究轮次预算耗尽，规划器开始收束")
        return None

    if event_type == TRUNCATED_EVENT:
        return _frame("error", seq, {"text": "实时推送拥塞，部分进度被跳过；刷新页面可回放完整进度"})

    # ---- RunManager 合成事件 ----
    if event_type == "run_status":
        status = _text(payload.get("status"), 32)
        return _frame("status", seq, {"status": status}) if status else None
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


def _task(
    event: str, seq: Any, payload: Mapping[str, Any], extra: str | dict
) -> SseFrame | None:
    """带 task_id 的帧统一从这里出；缺 task_id 的（异常生产者）静默丢弃。"""
    task = _text(payload.get("task_id"), 64)
    if not task:
        return None
    data = {"task": task}
    if isinstance(extra, str):
        data["title"] = extra
    else:
        data.update(extra)
    return _frame(event, seq, data)


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
