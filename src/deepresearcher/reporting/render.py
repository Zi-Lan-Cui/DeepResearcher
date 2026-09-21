"""最终报告渲染：按正文首现顺序编号，替换内部标记，生成参考来源表。

本模块是 [[cite:evidence_id]] → [来源N] 编号的唯一发生地；审阅通过后
由终检渲染节点调用一次。参考来源表携带 quote，保证最终交付物保留
可审计的原文引文，不因渲染而丢失 chunk。
"""

import re
from typing import cast
from urllib.parse import urlsplit

from deepresearcher.reporting.validation import _CITE_MARKER, _FENCED_CODE, _INLINE_CODE
from deepresearcher.schemas import Citation, ResearchDirectionResult, RunError, SupervisorProgress
from deepresearcher.state import ResearchState, section

# Writer 提示词禁止自写来源列表，但违令必须程序兜底：整段剥除（连同其中
# [[cite:…]] 标记，避免其抢占首现编号），编号权威只属于本模块的追加段。
_REFERENCE_HEADING = re.compile(
    r"^(#{2,3})[ \t]*(?:参考来源|引用来源|参考文献|引用列表|来源列表|参考文档|资料来源"
    r"|主要参考(?:资料|文献)?|引用(?:的)?来源|来源参考"
    r"|Sources?(?: and References?)?|References?|Bibliography)[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
_NEXT_HEADING = {
    2: re.compile(r"^#{1,2}[ \t]", re.MULTILINE),
    3: re.compile(r"^#{1,3}[ \t]", re.MULTILINE),
}


def _strip_reference_section(body: str) -> str:
    """剥除 Writer 草拟正文里自造的来源/参考文献小节（防御性兜底）。"""
    pieces: list[str] = []
    cursor = 0
    for heading in _REFERENCE_HEADING.finditer(body):
        before = body[cursor : heading.start()]
        if before.count("```") % 2:  # 处于代码围栏内：宁可漏剥也不误剥
            continue
        next_heading = _NEXT_HEADING[len(heading.group(1))].search(body, heading.end())
        end = next_heading.start() if next_heading else len(body)
        pieces.append(before)
        cursor = end
    pieces.append(body[cursor:])
    return "".join(pieces).strip()


def render_final_report(
    *,
    clarified_query: str,
    current_round: int,
    evidence_count: int,
    body: str,
    citations: list[Citation],
) -> str:
    """把 evidence_id 键草稿渲染为用户可见的最终报告。

    编号按正文首次出现顺序分配；未在正文出现的 citation 不进入参考表
    （Writer 的声明列表只是工作集提示，不是最终事实绑定）。
    """
    rendered_body, display_order, used_ids = _render_body_markers(_strip_reference_section(body))
    by_id = {item.id: item for item in citations}
    used_citations = [by_id[source_id] for source_id in display_order if source_id in by_id]
    display = {source_id: f"来源{index}" for index, source_id in enumerate(display_order, 1)}
    source_count = len({item.url for item in used_citations if item.url})

    lines = [
        "# 研究报告",
        "",
        "## 研究问题",
        clarified_query,
        "",
        f"> 已完成 {current_round} 轮研究，使用 {source_count} 个来源和 {evidence_count} 条 Evidence。",
        "> 以下为基于已验证 Evidence 的综合表述；关键事实以 [来源N] 标记可追溯到来源。",
        "",
        rendered_body,
    ]
    return "\n".join(lines) + _reference_list(display_order, display, by_id)


def _render_body_markers(body: str) -> tuple[str, list[str], set[str]]:
    """替换正文中的 [[cite:...]] 标记，返回 (渲染后正文, 首现顺序, 使用集合)。"""
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
            rendered.append(f"来源{display_order.index(source_id) + 1}")
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
    """生成带可审计引文的参考来源表：每行编号 + 标题/URL + 逐字引文。"""
    lines = ["", "## 参考来源"]
    for source_id in display_order:
        citation = by_id.get(source_id)
        if citation is None:
            continue  # 正文引用了无元数据的 id：降级为只有编号，不炸终检
        label = citation.title or urlsplit(citation.url).netloc or display[source_id]
        lines.append(f"- [{display[source_id]}] {label}: {citation.url}")
        if citation.quote:
            lines.append(f"  > 「{citation.quote}」")
    return "\n".join(lines)


def render_incomplete_report(state: ResearchState, reasons: list[str]) -> str:
    """研究未完成/写作失败路径的兜底报告；不补写任何研究结论。"""
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
        "> 已进入检索与证据验证流程，但当前未形成可交付的完整研究报告。",
        "> 因此不以模型内部知识补写研究结论。",
        "",
        "## 研究进度",
        f"已执行 {research.current_round} 轮，收集 {source_count} 个来源和 {evidence_count} 条 Evidence。",
    ]
    if task_results:
        lines.append("方向结果：")
        for task in task_results:
            status = (
                f"{task.execution_status}/{task.coverage_status}"
                if isinstance(task, ResearchDirectionResult)
                else f"{task.get('execution_status', 'unknown')}/{task.get('coverage_status', 'unknown')}"
            )
            direction = (
                task.research_direction
                if isinstance(task, ResearchDirectionResult)
                else str(task.get("research_direction", "未命名方向"))
            )
            lines.append(f"- [{status}] {direction}")
        gaps: list[str] = []
        for task in task_results:
            values = (
                task.remaining_gaps
                if isinstance(task, ResearchDirectionResult)
                else task.get("remaining_gaps", [])
            )
            gaps.extend(str(value) for value in values if str(value).strip())
        gaps.extend(str(value) for value in research.coverage_gaps if str(value).strip())
        if gaps:
            lines.append("未闭合缺口：")
            lines.extend(f"- {gap}" for gap in dict.fromkeys(gaps))
    lines.extend(
        [
            "",
            "## 已知阻塞",
        ]
    )
    lines.extend(f"- {reason}" for reason in reasons)
    return "\n".join(lines)


def render_error_report(state: ResearchState, error: RunError) -> str:
    """渲染面向用户的失败报告；异常原文只保留在日志和内部事件。"""
    reason = f"阶段 `{error.stage}` 执行失败，请稍后重试或重新发起。"
    return render_incomplete_report(state, [reason]) + "\n\n[流程状态：失败]"


def no_evidence_blockers(state: ResearchState) -> list[str]:
    """把 worker 的失败汇总为面向用户的阻塞信息，而非吞掉原始原因。"""
    task_results = state.get("task_results", [])
    failures: list[str] = []
    skipped: list[str] = []
    for task in task_results:
        if isinstance(task, ResearchDirectionResult):
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
