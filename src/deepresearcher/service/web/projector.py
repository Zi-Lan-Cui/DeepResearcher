"""事件 → 客户端安全帧 的纯函数投影。

未列入下列词汇表的 event_type 返回 None；payload 逐字段重建，不整包转发。
帧格式与前端共用；其中一部分帧由 service 侧合成：

  stage_open   阶段开始 {stage, title}
  plan         阶段旁白一行 {stage, text}
  stage_done   阶段结束与结论 {stage, status, text}
  tick         全局杂项一行
  stats        运行指标 {轮次, 任务, 证据计数}
  task_open    开一张任务卡 {task, title, state}（state=queued 表示在等并发槽）
  task_update  任务卡追加一条动作 {task, step}
  task_done    任务卡结束 {task, status, summary, finding, conclusion}
  status       运行状态变化（service 合成）
  headline     浓缩标题写入数据库（service 合成）
  error        需要用户知道的异常提示
  done         终态 {status, answer_mode, report_available}
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from deepresearcher.observability.events.names import EventName
from deepresearcher.routing import PREVIEW_CHANNELS, NodeName
from deepresearcher.schemas.limits import (
    EVENT_CONTENT_PREVIEW_CHARS,
    RUN_HEADLINE_MAX_CHARS,
    TASK_CONCLUSION_MAX_CHARS,
    TASK_FINDING_MAX_CHARS,
)

_STAGE_TITLES: dict[str, str] = {
    NodeName.ROUTER: "理解问题 · Router",
    NodeName.CLARIFY: "澄清范围 · Clarify",
    NodeName.QUICK_ANSWER: "即时回答",
    NodeName.SUPERVISOR: "研究规划 · Supervisor",
    NodeName.WRITER: "撰写报告 · Writer",
    NodeName.REVIEWER: "审阅报告 · Reviewer",
    NodeName.RENDER_FINAL_REPORT: "生成最终报告",
}

# 工具调用协议的字面标记：网关偶发把调用体当正文吐出（function=、parameter=、
# invoke name= 一类开标签，Claude 风格），这类回合的 text 不是旁白，整条丢弃。
_TOOL_MARKUP_RE = re.compile(r"<\s*(?:function[= ]|parameter[= ]|invoke\s)|<\s*antml:", re.I)

# source_read_skipped 的 reason_code 对应的用户可见措辞；未知码回退通用词，不透出机器码。
_SOURCE_SKIP_LABELS: dict[str, str] = {
    "empty_content": "无可读正文",
    "login_required": "需登录",
    "access_challenge": "访问被拦截",
    "fetch_failed": "网络不可达",
}


def _display_host(url: object) -> str:
    """URL → 展示用主机名：去掉协议、路径与查询串，只留可辨认的站点。"""
    try:
        host = urlsplit(str(url or "")).hostname or ""
    except ValueError:
        return ""
    return host.removeprefix("www.")[:48]


def _display_url(url: object, max_chars: int = 48) -> str:
    """URL → 展示用短地址：主机名加路径，超长时截去路径中段。

    同一站点的不同页面靠路径尾部区分，因此截断保留开头与结尾，去掉中间；
    尾段从完整的路径分隔处起，避免把一个词切成两截。
    """
    try:
        parts = urlsplit(str(url or ""))
    except ValueError:
        return ""
    host = (parts.hostname or "").removeprefix("www.")
    if not host:
        return ""
    path = parts.path.rstrip("/")
    if len(host) + len(path) <= max_chars:
        return host + path
    budget = max_chars - len(host) - 2
    if budget <= 0:
        return host[:max_chars]
    tail = path[-budget:]
    cut = tail.find("/")
    if cut > 0:
        tail = tail[cut:]
    return f"{host}/…{tail}"


@dataclass(frozen=True)
class SseFrame:
    event: str
    data: dict


def project_clarification(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    """clarification_requested 载荷 → 客户端可见的 {question, options}；空 question 不成帧。

    截断上限为 options 最多 3 条、每条 120 字，question 最多 500 字。这是敏感
    文案的白名单出口，SSE 与 REST 两条通道共用此一处，改动不得另起一份副本。
    """
    question = _text(payload.get("question"), 500)
    raw_options = payload.get("options")
    options = (
        [_text(item, 120) for item in raw_options[:3] if _text(item, 120)]
        if isinstance(raw_options, list)
        else []
    )
    return {"question": question, "options": options} if question else None


def project_run_error(payload: Any) -> dict[str, str] | None:
    """落库的 RunError → 详情页可见的中断三项；两项正文都缺则不成帧。"""
    if not isinstance(payload, Mapping):
        return None
    code = _text(payload.get("code"), 64)
    message = _text(payload.get("message"), 200)
    if not (code or message):
        return None
    stage = _text(payload.get("stage"), 64)
    return {
        "stage": stage,
        "stage_label": _STAGE_TITLES.get(stage, stage) if stage else "",
        "code": code,
        "message": message,
    }


# 「弃权」是审阅失败的旧叫法；旧行仍这么写，出口统一成新词，前端只认一个。
_REVIEW_STATUS_ALIASES = {"abstained": "failed"}


def project_review(payload: Any) -> dict[str, Any] | None:
    """落库的 ReviewProgress → 详情页可见的轮次与意见；未进入审阅的运行返回 None。"""
    if not isinstance(payload, Mapping):
        return None
    status = _text(payload.get("status"), 16)
    if not status or status == "pending":
        return None
    raw_issues = payload.get("issues")
    issues = [
        _text(item.get("reason"), 160)
        for item in (raw_issues if isinstance(raw_issues, list) else [])
        if isinstance(item, Mapping) and _text(item.get("reason"), 160)
    ][:3]
    return {
        "status": _REVIEW_STATUS_ALIASES.get(status, status),
        "attempts": _int(payload.get("attempts")),
        "feedback": _text(payload.get("feedback"), 300),
        # 旧行把这一栏写成 abstain_reason；换词前的 run 不该因此丢掉失败原因。
        "failure_reason": _text(payload.get("failure_reason"), 200)
        or _text(payload.get("abstain_reason"), 200),
        "issues": issues,
    }


def project(record: Mapping[str, Any]) -> SseFrame | None:
    """引擎/合成事件 dict → 客户端帧；不在白名单内的一律 None。

    帧里统一附加 dedup_key 供客户端识别重放，投影分支不必各自处理。
    """
    frame = _project(record)
    if frame is None:
        return None
    dedup_key = record.get("dedup_key")
    if isinstance(dedup_key, str) and dedup_key:
        frame.data["dedup_key"] = dedup_key
    return frame


def _project(record: Mapping[str, Any]) -> SseFrame | None:
    """按事件类型投影为客户端帧，不处理去重键；未登记的类型返回 None。"""
    event_type = record.get("event_type")
    if not isinstance(event_type, str):
        return None
    projector = _PROJECTORS.get(event_type)
    if projector is None:
        return None
    payload = record.get("payload")
    if not isinstance(payload, Mapping):
        payload = {}
    return projector(record.get("seq"), record.get("node"), payload)


# 阶段生命周期
def _node_started(seq: Any, node: Any, payload: Mapping[str, Any]) -> SseFrame | None:
    """未登记的节点名退回一行杂项，不新开阶段框。"""
    if not isinstance(node, str) or node not in _STAGE_TITLES:
        return _tick(seq, f"{node} 开始" if isinstance(node, str) else None)
    return _frame("stage_open", seq, {"stage": node, "title": _STAGE_TITLES[node]})


def _node_completed(seq: Any, node: Any, payload: Mapping[str, Any]) -> SseFrame | None:
    """只关已开的阶段框；未登记的节点没有框可关，也不该冒出一行杂项。"""
    if isinstance(node, str) and node in _STAGE_TITLES:
        return _frame(
            "stage_done",
            seq,
            {"stage": node, "status": "done", "text": _stage_conclusion(node, payload)},
        )
    return None


def _node_failed(seq: Any, node: Any, payload: Mapping[str, Any]) -> SseFrame | None:
    """未登记的节点退回全局错误行，因为它的失败没有对应的阶段框。

    内部异常文本不外发，完整信息留在 RunEvent 与日志。
    """
    if isinstance(node, str) and node in _STAGE_TITLES:
        return _frame(
            "stage_done",
            seq,
            {"stage": node, "status": "failed", "text": f"{_STAGE_TITLES[node]} 执行失败"},
        )
    return _frame(
        "error", seq, {"text": f"{node if isinstance(node, str) else '某阶段'} 阶段执行失败"}
    )


def _node_cancelled(seq: Any, node: Any, payload: Mapping[str, Any]) -> SseFrame | None:
    """已开框的阶段必须发 cancelled 的 stage_done，只发 tick 会让框停在运行中。"""
    if isinstance(node, str) and node in _STAGE_TITLES:
        return _frame(
            "stage_done",
            seq,
            {"stage": node, "status": "cancelled", "text": "该阶段已取消"},
        )
    return _tick(seq, "该阶段已取消")


# Supervisor 决策旁白：plan 帧都带 stage 归属
def _delegate_completed(seq: Any, node: Any, payload: Mapping[str, Any]) -> SseFrame | None:
    """只在轮次预算耗尽时留一行；其余派发结果由任务卡承载。"""
    if str(payload.get("status", "")) == "blocked":
        return _plan(seq, NodeName.SUPERVISOR, "研究轮次预算耗尽，停止派发新任务")
    return None


def _task_merged(seq: Any, node: Any, payload: Mapping[str, Any]) -> SseFrame | None:
    """在途任务回到工作台的一行旁白，区分「进程重启丢失」与「正常合并」。"""
    title = _text(payload.get("display_title"), 40) or _text(payload.get("task_id"), 40)
    if payload.get("merged_via") == "lost":
        return _plan(seq, NodeName.SUPERVISOR, f"任务「{title}」因进程重启丢失，可按需重派")
    text = "任务「{}」已回到工作台（证据 {}）".format(title, _int(payload.get("evidence_count")))
    running = payload.get("running_task_ids") or []
    if running:
        text += f"，还有 {len(running)} 个任务在途"
    return _plan(seq, NodeName.SUPERVISOR, text)


def _supervisor_model_turn(seq: Any, node: Any, payload: Mapping[str, Any]) -> SseFrame | None:
    """模型回合的思考摘要；网关把工具调用体当正文吐出时整条丢弃。

    那不是旁白，工具照常执行，用户不该在时间线里读到调用协议。
    """
    thought = _text(payload.get("content_preview"), EVENT_CONTENT_PREVIEW_CHARS)
    if thought and _TOOL_MARKUP_RE.search(thought):
        return None
    return _plan(seq, NodeName.SUPERVISOR, thought) if thought else None


# 任务卡：Supervisor 块内子项
def _research_task_queued(seq: Any, node: Any, payload: Mapping[str, Any]) -> SseFrame | None:
    """受理即建卡：任务已入队但可能还在等并发槽，state 让前端区分排队与进行中。"""
    title = _text(payload.get("title"), 40) or _text(payload.get("question"), 140)
    return _task("task_open", seq, payload, {"title": title, "state": "queued"})


def _research_task_started(seq: Any, node: Any, payload: Mapping[str, Any]) -> SseFrame | None:
    """优先用派发时给出的短题；旧事件没有 title，回退到截断后的 question。"""
    title = _text(payload.get("title"), 40) or _text(payload.get("question"), 140)
    return _task("task_open", seq, payload, {"title": title})


def _research_task_completed(seq: Any, node: Any, payload: Mapping[str, Any]) -> SseFrame | None:
    """完成帧带三行：计数摘要、卡面摘要、展开后的完整结论。"""
    summary = "证据 {} · 来源 {}".format(
        _int(payload.get("evidence_count")), _int(payload.get("source_count"))
    )
    return _task(
        "task_done",
        seq,
        payload,
        {
            "status": _text(payload.get("execution_status"), 16) or "done",
            "summary": summary,
            "finding": _text(payload.get("finding"), TASK_FINDING_MAX_CHARS),
            "conclusion": _text(payload.get("conclusion"), TASK_CONCLUSION_MAX_CHARS),
        },
    )


def _research_task_failed(seq: Any, node: Any, payload: Mapping[str, Any]) -> SseFrame | None:
    return _task("task_done", seq, payload, {"status": "failed", "summary": "研究未成功"})


def _task_evidence_added(seq: Any, node: Any, payload: Mapping[str, Any]) -> SseFrame | None:
    """本次提交的审计：accepted=0 表示全被退回修正，不是新增。"""
    accepted = _int(payload.get("accepted_count"))
    rejected = _int(payload.get("rejected_count")) + _int(payload.get("duplicate_count"))
    if accepted:
        return _step(
            seq,
            payload,
            "提交",
            "证据",
            f"{accepted} 条通过",
            note=f"{rejected} 条退回" if rejected else "",
        )
    if rejected:
        return _step(seq, payload, "提交", "证据", note="未通过校验")
    return None


def _source_document_registered(seq: Any, node: Any, payload: Mapping[str, Any]) -> SseFrame | None:
    """材料缓存命中要标成「复用」而不是「读取」。

    同一来源被两个任务读取时，一律写「读取」会让用户以为重复抓取了两次。
    """
    raw = payload.get("final_url") or payload.get("requested_url")
    shown = _display_url(raw) or "一份来源"
    cached = bool(payload.get("material_cache_hit"))
    return _step(seq, payload, "复用" if cached else "读取", shown, title=str(raw or ""))


def _source_read_failed(seq: Any, node: Any, payload: Mapping[str, Any]) -> SseFrame | None:
    """读取失败与跳过在左列分开：前者是异常，后者是按条件放弃。"""
    raw = payload.get("url")
    return _step(
        seq,
        payload,
        "失败",
        _display_host(raw) or "一份来源",
        note="读取失败",
        title=str(raw or ""),
    )


def _source_read_skipped(seq: Any, node: Any, payload: Mapping[str, Any]) -> SseFrame | None:
    """未知 reason_code 回退通用词，不透出机器码。"""
    label = _SOURCE_SKIP_LABELS.get(str(payload.get("reason_code")), "不可读")
    raw = payload.get("url")
    return _step(
        seq,
        payload,
        "跳过",
        _display_host(raw) or "一份来源",
        note=label,
        title=str(raw or ""),
    )


def _task_search_completed(seq: Any, node: Any, payload: Mapping[str, Any]) -> SseFrame | None:
    """批次行只报逐查询行覆盖不到的信息。

    逐查询结果由 search_query_completed 各占一行，候选总数与失败数都能由这些行加出，
    再报一遍只是重复。批次行只在两种情况下不可省：整批没有逐查询行可发（熔断、预检），
    以及部分查询命中进程内缓存——缓存查询不发逐查询事件，界面上原本没有它的位置。
    """
    status = str(payload.get("status") or "")
    if status and status != "completed":
        return _step(seq, payload, "检索", "本批查询", note="失败")
    cached = _int(payload.get("cache_hit_count"))
    if not cached:
        return None
    return _step(
        seq,
        payload,
        "复用",
        f"{cached} 个查询",
        f"{_int(payload.get('candidate_count'))} 条候选",
        note="命中缓存",
    )


def _search_query_completed(seq: Any, node: Any, payload: Mapping[str, Any]) -> SseFrame | None:
    """一条查询一行；没搜到结果时右列留空，原因进括注，右列只放同构计数。"""
    count = _int(payload.get("candidate_count"))
    return _step(
        seq,
        payload,
        "检索",
        f"「{_text(payload.get('query'), 60)}」",
        f"{count} 条候选" if count else "",
        note="" if count else "未见相关来源",
    )


def _search_query_failed(seq: Any, node: Any, payload: Mapping[str, Any]) -> SseFrame | None:
    query = _text(payload.get("query"), 60)
    return _step(seq, payload, "检索", f"「{query}」", note="失败")


# 指标与杂项
def _research_round_completed(seq: Any, node: Any, payload: Mapping[str, Any]) -> SseFrame | None:
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


def _text_delta(seq: Any, node: Any, payload: Mapping[str, Any]) -> SseFrame | None:
    """流式正文帧，通道名与总线开关读同一份 routing.PREVIEW_CHANNELS。

    生产者是 RunExecutor 把 astream(subgraphs=True) 的 messages 按命名空间路由；
    这里是投影侧的再次校验，未登记通道一律丢弃。
    """
    channel = _text(payload.get("channel"), 24)
    text = _text(payload.get("text"), 200)
    if channel not in PREVIEW_CHANNELS or not text:
        return None
    return SseFrame(event=EventName.TEXT_DELTA, data={"channel": channel, "text": text})


# service 合成的帧
def _run_headline_updated(seq: Any, node: Any, payload: Mapping[str, Any]) -> SseFrame | None:
    headline = _text(payload.get("headline"), RUN_HEADLINE_MAX_CHARS)
    return _frame("headline", seq, {"text": headline}) if headline else None


def _run_status(seq: Any, node: Any, payload: Mapping[str, Any]) -> SseFrame | None:
    status = _text(payload.get("status"), 32)
    return _frame("status", seq, {"status": status}) if status else None


def _clarification_requested(seq: Any, node: Any, payload: Mapping[str, Any]) -> SseFrame | None:
    clarification = project_clarification(payload)
    return _frame("clarification", seq, clarification) if clarification else None


def _run_done(seq: Any, node: Any, payload: Mapping[str, Any]) -> SseFrame | None:
    return _frame(
        "done",
        seq,
        {
            "status": _text(payload.get("status"), 32),
            "answer_mode": _text(payload.get("answer_mode"), 32),
            "report_available": bool(payload.get("report_available")),
        },
    )


#: 事件类型到投影函数。新增一种事件只在这里登记一行，投影主体不随之改动。
_PROJECTORS: dict[str, Callable[[Any, Any, Mapping[str, Any]], SseFrame | None]] = {
    EventName.NODE_STARTED: _node_started,
    EventName.NODE_COMPLETED: _node_completed,
    EventName.NODE_FAILED: _node_failed,
    EventName.NODE_CANCELLED: _node_cancelled,
    EventName.DELEGATE_COMPLETED: _delegate_completed,
    EventName.TASK_MERGED: _task_merged,
    EventName.SUPERVISOR_MODEL_TURN: _supervisor_model_turn,
    EventName.RESEARCH_TASK_QUEUED: _research_task_queued,
    EventName.RESEARCH_TASK_STARTED: _research_task_started,
    EventName.RESEARCH_TASK_COMPLETED: _research_task_completed,
    EventName.RESEARCH_TASK_FAILED: _research_task_failed,
    EventName.TASK_EVIDENCE_ADDED: _task_evidence_added,
    EventName.SOURCE_DOCUMENT_REGISTERED: _source_document_registered,
    EventName.SOURCE_READ_FAILED: _source_read_failed,
    EventName.SOURCE_READ_SKIPPED: _source_read_skipped,
    EventName.TASK_SEARCH_COMPLETED: _task_search_completed,
    EventName.SEARCH_QUERY_COMPLETED: _search_query_completed,
    EventName.SEARCH_QUERY_FAILED: _search_query_failed,
    EventName.RESEARCH_ROUND_COMPLETED: _research_round_completed,
    EventName.TEXT_DELTA: _text_delta,
    EventName.RUN_HEADLINE_UPDATED: _run_headline_updated,
    EventName.RUN_STATUS: _run_status,
    EventName.CLARIFICATION_REQUESTED: _clarification_requested,
    EventName.RUN_DONE: _run_done,
}


def _stage_conclusion(node: str, payload: Mapping[str, Any]) -> str:
    """生成阶段完成时用户可见的结论文本；只读白名单键，各键均已截断到有界长度。"""
    if node == NodeName.ROUTER:
        route = str(payload.get("route", ""))
        label = {"deep_research": "进入深度研究", "quick_answer": "按即时回答处理"}.get(
            route, route
        )
        reason = _text(payload.get("route_reason"), 120)
        return "；".join(part for part in (label, reason) if part)
    if node == NodeName.CLARIFY:
        # 澄清结论一行
        return _text(payload.get("intent_summary"), 160) or _text(
            payload.get("clarified_query"), 300
        )
    if node == NodeName.SUPERVISOR:
        return "规划完成 · 第 {} 轮 · 本次新增证据 {}".format(
            _int(payload.get("current_round")), _int(payload.get("evidences_count"))
        )
    if node == NodeName.WRITER:
        return {
            "completed": "草稿通过校验",
            "exhausted": "写作轮次耗尽",
            "failed": "写作失败",
        }.get(str(payload.get("writer_status", "")), "")
    if node == NodeName.REVIEWER:
        label = {
            "approved": "审阅通过",
            "rejected": "审阅要求修改",
            "failed": "审阅失败",
        }.get(str(payload.get("review_status", "")), "")
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


def _step(
    seq: Any,
    payload: Mapping[str, Any],
    act: str,
    obj: str,
    res: str = "",
    *,
    note: str = "",
    title: str = "",
) -> SseFrame | None:
    """任务卡的一行动作：左列动作词、中间对象、右列计数，括注与悬停地址按需附加。

    三段分开给，是因为它们的排版职责不同：act 定宽便于纵向对齐，res 只放同构计数
    以便右对齐成一列，note 是非计数的原因说明，跟在对象后面。
    """
    step = {"act": act, "obj": obj, "res": res}
    if note:
        step["note"] = note
    if title:
        step["title"] = title
    return _task("task_update", seq, payload, {"step": step})


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
