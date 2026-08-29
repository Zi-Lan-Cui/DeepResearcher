"""事件 → 客户端安全帧 的纯函数投影。

**默认拒绝**：只处理映射表里显式列出的 event_type，未知类型返回 None——
引擎将来新增事件不会悄悄把 payload 漏到浏览器。每一帧都逐字段新建 dict，
绝不整包转发 payload；``error``/``*_preview``/``prompt``/``queries`` 全列表等
敏感键在代码层面就没有进入路径。

SSE 帧词汇表（与前端、RunManager 合成事件共用）：

- ``tick``   进度文案行
- ``status`` 运行状态变化（RunManager 合成）
- ``error``  需要用户知道的异常提示（仅安全文案）
- ``done``   终态，data 带 {status, answer_mode, report_available}
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from deepsearch_agent.routing import NodeName
from deepsearch_agent.schemas import StopReason

TRUNCATED_EVENT = "stream_truncated"

_MODEL_TURN_SUFFIX = "_model_turn"

# 中文播报文案。改动无需回归引擎——这是展示层词汇，与事件生产者解耦。
_NODE_NARRATION: dict[str, str] = {
    NodeName.ROUTER: "正在理解问题…",
    NodeName.CLARIFY: "正在澄清研究范围…",
    NodeName.QUICK_ANSWER: "正在准备即时回答…",
    NodeName.SUPERVISOR: "正在拆解研究任务…",
    NodeName.WRITER: "正在撰写报告…",
    NodeName.REFLECTION: "正在审阅报告…",
    NodeName.RENDER_FINAL_REPORT: "正在生成最终报告…",
}

_TURN_AGENT_LABELS: dict[str, str] = {
    "supervisor": "研究规划",
    "writer": "报告撰写",
    "researcher": "方向检索",
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

    if event_type == "direction_search_completed":
        direction = _text(payload.get("research_direction"), 60)
        count = _int(payload.get("candidate_count"))
        return _tick(seq, f"检索『{direction}』完成：{count} 条候选来源")
    if event_type == "research_round_completed":
        return _tick(
            seq,
            "第 {} 轮研究完成：方向 {}/{}，新增证据 {}（累计 {}）".format(
                _int(payload.get("round")),
                _int(payload.get("completed_tasks")),
                _int(payload.get("task_count")),
                _int(payload.get("evidence_added")),
                _int(payload.get("total_evidence_count")),
            ),
        )
    if event_type == "research_stopped":
        return _tick(seq, f"研究提前结束：{_stop_reason_text(payload.get('reason'))}")
    if event_type in {"source_fetch_completed", "source_reader_completed"}:
        return _tick(seq, "来源读取完成")
    if event_type == "evidence_chunk_completed":
        return _tick(seq, f"证据抽取：候选 {_int(payload.get('candidate_count'))} 条")
    if event_type == "delegate_completed":
        # 规划器被静默消化的工具调用（重复方向/预算闸口）——让“空轮次”在直播里可见。
        status = str(payload.get("status", ""))
        if status == "skipped":
            return _tick(seq, "发现重复研究方向，已跳过并调整计划")
        if status == "blocked":
            return _tick(seq, "研究轮次预算耗尽，规划器开始收束")
        return None
    if event_type == "writer_draft_ready":
        return _tick(seq, "报告草稿完成，进入审阅")

    if event_type.endswith(_MODEL_TURN_SUFFIX):
        agent = event_type[: -len(_MODEL_TURN_SUFFIX)]
        label = _TURN_AGENT_LABELS.get(agent)
        if label is None:
            return None
        return _tick(seq, f"{label}中（第 {_int(payload.get('turn'))} 轮）")

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
