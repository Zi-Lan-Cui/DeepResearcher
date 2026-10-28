"""最终报告渲染：按正文首现顺序编号，替换内部标记，生成证据来源表。"""

import re
from typing import cast
from urllib.parse import urlsplit

from deepresearcher.evidence.models import normalize_title_whitespace
from deepresearcher.reporting.validation import _CITE_MARKER, _FENCED_CODE, _INLINE_CODE
from deepresearcher.schemas import Citation, RunError, SupervisorProgress, TaskResult
from deepresearcher.state import RunState, section

# 检查模型是否在正文生成了证据来源小节，若有则移除；避免用户看到重复的来源列表。
_REFERENCE_HEADING = re.compile(
    r"^(#{2,3})[ \t]*(?:证据来源|参考来源|引用来源|参考文献|引用列表|来源列表|参考文档|资料来源"
    r"|主要参考(?:资料|文献)?|引用(?:的)?来源|来源参考"
    r"|Sources?(?: and References?)?|References?|Bibliography)[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
_NEXT_HEADING = {
    2: re.compile(r"^#{1,2}[ \t]", re.MULTILINE),
    3: re.compile(r"^#{1,3}[ \t]", re.MULTILINE),
}


def _strip_reference_section(body: str) -> str:
    """移除 Writer 草拟正文里自造的来源/参考文献小节（防御性处理）。"""
    pieces: list[str] = []
    cursor = 0
    for heading in _REFERENCE_HEADING.finditer(body):
        before = body[cursor : heading.start()]
        if before.count("```") % 2:  # 处于代码围栏内：宁可漏删也不误删
            continue
        next_heading = _NEXT_HEADING[len(heading.group(1))].search(body, heading.end())
        end = next_heading.start() if next_heading else len(body)
        pieces.append(before)
        cursor = end
    pieces.append(body[cursor:])
    return "".join(pieces).strip()


def render_final_report(
    *,
    body: str,
    citations: list[Citation],
    title: str | None = None,
) -> str:
    """把 evidence_id 键草稿渲染为用户可见的最终报告。

    编号按正文首次出现顺序分配；未在正文出现的 citation 不进入证据来源表
    （Writer 的声明列表只是工作集提示，不是最终事实绑定）。
    研究问题与轮次统计由页面展示，不写入报告正文。
    """
    rendered_body, display_order, _used_ids = _render_body_markers(_strip_reference_section(body))
    by_id = {item.id: item for item in citations}
    display = {source_id: f"证据{index}" for index, source_id in enumerate(display_order, 1)}

    # 文章结构：H1 是 Writer 拟的标题，为空时用默认标题。
    heading = (title or "").strip() or "研究报告"
    return f"# {heading}\n\n{rendered_body}" + _reference_list(display_order, display, by_id)


def _render_body_markers(body: str) -> tuple[str, list[str], set[str]]:
    """替换正文中的 [[cite:...]] 标记，返回 (渲染后正文， 首现顺序， 使用集合)。"""
    display_order: list[str] = []
    used_ids: set[str] = set()

    def render_marker(match: re.Match[str]) -> str:
        raw_ids = [
            item.strip().strip("[]")
            for item in re.split(r"[,，、]", match.group("sources"))
            if item.strip()
        ]
        source_ids = list(dict.fromkeys(raw_ids))
        rendered: list[str] = []
        for source_id in source_ids:
            if source_id not in used_ids:
                display_order.append(source_id)
                used_ids.add(source_id)
            rendered.append(f"证据{display_order.index(source_id) + 1}")
        return " ".join(f"[{item}]" for item in rendered)

    def replace_in_prose(prose: str) -> str:
        pieces: list[str] = []
        cursor = 0
        for inline in _INLINE_CODE.finditer(prose):
            pieces.append(_CITE_MARKER.sub(render_marker, prose[cursor : inline.start()]))
            pieces.append(inline.group(0))
            cursor = inline.end()
        pieces.append(_CITE_MARKER.sub(render_marker, prose[cursor:]))
        return "".join(pieces)

    parts: list[str] = []
    cursor = 0
    for fenced in _FENCED_CODE.finditer(body):
        parts.append(replace_in_prose(body[cursor : fenced.start()]))
        parts.append(fenced.group(0))
        cursor = fenced.end()
    parts.append(replace_in_prose(body[cursor:]))
    return "".join(parts).strip(), display_order, used_ids


def _reference_list(
    display_order: list[str],
    display: dict[str, str],
    by_id: dict[str, Citation],
) -> str:
    """生成带可审计引文的证据来源表：每行编号 + 标题/URL + 逐字引文。"""
    lines = ["", "## 证据来源"]
    for source_id in display_order:
        citation = by_id.get(source_id)
        if citation is None:
            continue  # 正文引用了无元数据的 id：降级为只显示编号，不中断终检
        label = citation.title or urlsplit(citation.url).netloc or display[source_id]
        # 表行按行消费，标题或引文中残留的换行会把一条记录劈断（前端按行解析即失配），
        # 生成侧统一做折行处理，存量数据重渲染同样受护。
        lines.append(
            f"- [{display[source_id]}] {normalize_title_whitespace(label)}: {citation.url}"
        )
        if citation.quote:
            lines.append(f"  > 「{normalize_title_whitespace(citation.quote)}」")
    return "\n".join(lines)


# 部分报告里的任务只是诊断线索，不是交付主体：短题化后逐条列出。
_TASK_SUMMARY_MAX_CHARS = 40


def _task_field(task: object, name: str, default: object = None) -> object:
    """TaskResult 与 checkpoint 恢复态的 dict 两种形态统一取值。"""
    if isinstance(task, TaskResult):
        return getattr(task, name, default)
    return cast(dict[str, object], task).get(name, default)


def _summarize(text: object, max_chars: int) -> str:
    flattened = " ".join(str(text).split())
    if len(flattened) <= max_chars:
        return flattened
    return flattened[: max_chars - 1] + "…"


def render_incomplete_report(state: RunState, reasons: list[str]) -> str:
    """研究未完成/写作失败路径的降级报告；不补写任何研究结论。

    零证据路径塌缩为诊断卡（原因与恢复方式）；有证据的部分报告只列任务短题与状态，
    不复述方向结论——报告只认 Supervisor 给的任务书，方向内容留在工作台。
    """
    evidences = state.get("evidences", [])
    task_results = state.get("task_results", [])
    research = section(state, "supervisor", SupervisorProgress)
    evidence_count = len(evidences)
    source_count = len({item.source_url for item in evidences if item.source_url})
    lines = [
        "# 研究未完成",
        "",
        "## 研究问题",
        str(state.get("clarified_query") or state.get("query") or "（未提供研究问题）"),
        "",
        "## 研究进度",
        f"已执行 {research.current_round} 轮，收集 {source_count} 个来源和 {evidence_count} 条证据。",
        "",
        "## 阻塞原因",
    ]
    lines.extend(f"- {reason}" for reason in reasons)
    if evidence_count == 0:
        # 0 证据时任务级缺口全是"无法获取"类复读，列出只会淹没真正的原因。
        if any(bool(_task_field(task, "provider_exhausted")) for task in task_results):
            lines.append("- 搜索服务额度或凭据不可用，属系统性故障，与问题本身无关。")
        lines.extend(
            [
                "",
                "本次研究没有收集到任何可验证的证据，因此不给出结论，也不以模型内部知识补写。",
                "恢复方式：补充或更换搜索服务凭据（env/.env）后重新提交该问题；本次运行不保留部分结果。",
            ]
        )
        return "\n".join(lines)
    if task_results:
        lines.extend(["", "### 任务结果"])
        for task in task_results:
            status = f"{_task_field(task, 'execution_status', 'unknown')}/{_task_field(task, 'coverage_status', 'unknown')}"
            task_title = _summarize(
                _task_field(task, "research_direction", "未命名任务"), _TASK_SUMMARY_MAX_CHARS
            )
            task_evidence = _task_field(task, "evidence_count", 0)
            lines.append(f"- {task_title}（{status}，证据 {task_evidence} 条）")
    return "\n".join(lines)


def render_error_report(state: RunState, error: RunError) -> str:
    """渲染面向用户的失败报告；异常原文与阶段标识只保留在日志和内部事件。"""
    del error
    return render_incomplete_report(state, ["研究流程中途失败，未能完成本次交付；请稍后重新提交。"])


def no_evidence_blockers(state: RunState) -> list[str]:
    """把 worker 的失败汇总为面向用户的阻塞信息，同时保留原始失败原因。"""
    task_results = state.get("task_results", [])
    failures: list[str] = []
    skipped: list[str] = []
    for task in task_results:
        if isinstance(task, TaskResult):
            failures.extend(str(error) for error in task.failures if error)
            skipped.extend(str(reason) for reason in task.skip_reasons if reason)
        else:
            raw_task = cast(dict[str, object], task)
            failures.extend(
                str(error) for error in cast(list[object], raw_task.get("failures", [])) if error
            )
            skipped.extend(
                str(reason)
                for reason in cast(list[object], raw_task.get("skip_reasons", []))
                if reason
            )
    if any("Insufficient Balance" in error for error in failures):
        return ["证据抽取模型调用失败：API 余额不足（Insufficient Balance）。"]
    blockers: list[str] = []
    if failures:
        examples = list(dict.fromkeys(failures))[:3]
        blockers.append(
            f"{len(failures)} 次候选来源读取或证据抽取失败；代表性原因：" + "；".join(examples)
        )
    if skipped:
        counts = {reason: skipped.count(reason) for reason in sorted(set(skipped))}
        blockers.append(
            "候选来源未产生可验证 Evidence："
            + "，".join(f"{reason} × {count}" for reason, count in counts.items())
        )
    return blockers or ["未获取到可验证 Evidence，不能形成研究结论。"]
