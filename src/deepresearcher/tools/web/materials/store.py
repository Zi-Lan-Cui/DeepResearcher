"""研究材料存储协议与纯函数读取逻辑。"""

from typing import Protocol

from deepresearcher.tools.web.documents import (
    DocumentGrepMatch,
    DocumentOutlineItem,
    DocumentReadRange,
    DocumentRef,
    GrepPatternsResult,
    GrepSegmentResult,
)
from deepresearcher.tools.web.materials.models import SearchResultSet
from deepresearcher.vocab import RETRIEVAL_ORIGIN_FETCH


class ResearchMaterialStore(Protocol):
    """搜索目录与网页正文的统一临时存储边界。"""

    async def put_search_results(self, result_set: SearchResultSet) -> None: ...

    async def get_search_results(self, run_id: str, search_id: str) -> SearchResultSet: ...

    async def put(
        self,
        *,
        text: str,
        title: str,
        source_url: str,
        published_at: str = "",
        retrieval_method: str = RETRIEVAL_ORIGIN_FETCH,
        support_ceiling: str = "direct",
        token_count: int = 0,
        outline: list[DocumentOutlineItem] | None = None,
        fetch_key: str = "",
    ) -> DocumentRef: ...

    async def resolve_fetch(self, fetch_key: str) -> DocumentRef | None: ...

    async def get(self, document_id: str) -> DocumentRef: ...

    async def text(self, document_id: str) -> str: ...

    async def read(
        self,
        document_id: str,
        ranges: list[tuple[int, int]],
        *,
        max_lines: int,
        max_chars: int,
    ) -> list[DocumentReadRange]: ...

    async def grep_patterns(
        self,
        document_id: str,
        patterns: list[str],
        *,
        context_lines: int,
        segments: list[tuple[int, int]],
        max_matches_per_segment: int,
        max_chars: int,
        max_line_chars: int,
    ) -> GrepPatternsResult: ...

    async def close(self) -> None: ...


_READ_LINE_OMITTED_TEMPLATE = "……[本行其余 {omitted} 字符未显示]"
_GREP_LINE_OMITTED_TEMPLATE = "……[本行其余 {omitted} 字符未显示，整行请用 ReadDocument 补读]"
_GREP_WINDOW_OMITTED_TEMPLATE = (
    "……[本窗口其余 {lines} 行（L{start}-L{end}）未显示，请用 ReadDocument 补读]"
)


def _clip_line(text: str, budget: int, omitted_template: str) -> str:
    """渲染行超出字符预算时裁切，并标注剩余字符数。"""

    if len(text) <= budget:
        return text
    marker_bound = omitted_template.format(omitted=len(text))
    visible = max(0, budget - len(marker_bound))
    return text[:visible] + omitted_template.format(omitted=len(text) - visible)


def read_lines(
    lines: list[str],
    ranges: list[tuple[int, int]],
    *,
    max_lines: int,
    max_chars: int,
) -> list[DocumentReadRange]:
    """对已加载文本做有界行号读取；存储后端共用同一语义。

    整行超预算时裁切标注而非静默截断——静默截断会被模型当成完整原文引用。
    """

    results: list[DocumentReadRange] = []
    used_lines = 0
    used_chars = 0
    for requested_start, requested_end in ranges:
        if requested_end < requested_start:
            raise ValueError("end_line 不能小于 start_line")
        start = min(requested_start, len(lines) + 1)
        end = min(requested_end, len(lines))
        if start > end:
            continue
        remaining_lines = max_lines - used_lines
        remaining_chars = max_chars - used_chars
        if remaining_lines <= 0 or remaining_chars <= 0:
            break
        rendered: list[str] = []
        rendered_chars = 0
        for number in range(start, min(end, start + remaining_lines - 1) + 1):
            item = f"L{number}: {lines[number - 1]}"
            separator_chars = 1 if rendered else 0
            if rendered_chars + separator_chars + len(item) > remaining_chars:
                if rendered:
                    break
                item = _clip_line(item, remaining_chars, _READ_LINE_OMITTED_TEMPLATE)
            rendered.append(item)
            rendered_chars += separator_chars + len(item)
            if rendered_chars >= remaining_chars:
                break
        if not rendered:
            break
        content = "\n".join(rendered)
        results.append(
            DocumentReadRange(
                start_line=start,
                end_line=start + len(rendered) - 1,
                content=content,
            )
        )
        used_lines += len(rendered)
        used_chars += len(content)
    return results


def _render_grep_window(
    lines: list[str],
    window: tuple[int, int],
    *,
    used_chars: int,
    max_chars: int,
    line_budget: int,
) -> DocumentGrepMatch | None:
    """按剩余页预算渲染单个命中窗口；一行都放不下时返回 None，由调用方停止发页。"""

    start, end = window
    rendered: list[str] = []
    window_chars = 0
    cut_line: int | None = None
    for line_number in range(start, end + 1):
        item = _clip_line(
            f"L{line_number}: {lines[line_number - 1]}",
            line_budget,
            _GREP_LINE_OMITTED_TEMPLATE,
        )
        separator_chars = 1 if rendered else 0
        if used_chars + window_chars + separator_chars + len(item) > max_chars:
            cut_line = line_number
            break
        rendered.append(item)
        window_chars += separator_chars + len(item)
    if not rendered:
        return None
    emitted_line_count = len(rendered)
    tail_omitted_lines = end - cut_line + 1 if cut_line is not None else 0
    if tail_omitted_lines:
        note = _GREP_WINDOW_OMITTED_TEMPLATE.format(
            lines=tail_omitted_lines, start=cut_line, end=end
        )
        if used_chars + window_chars + 1 + len(note) <= max_chars:
            rendered.append(note)
            window_chars += 1 + len(note)
    return DocumentGrepMatch(
        hit_patterns=[],
        start_line=start,
        end_line=start + emitted_line_count - 1,
        content="\n".join(rendered),
        tail_omitted_lines=tail_omitted_lines,
    )


def grep_patterns(
    lines: list[str],
    patterns: list[str],
    *,
    context_lines: int,
    segments: list[tuple[int, int]],
    max_matches_per_segment: int,
    max_chars: int,
    max_line_chars: int,
) -> GrepPatternsResult:
    """多关键词字面检索：命中按文档行号归并成一条流，segments 从流上切窗。

    每词命中窗口各自去重后按行号合并，offset 恒指归并流位置。字符预算在整份结果内
    共享，耗尽后其余段 returned=0 并置 has_more。超长行裁切并标注，被裁行不属于后段。
    """

    folded_lines = [line.casefold() for line in lines]
    pattern_totals: dict[str, int] = {}
    tags_by_window: dict[tuple[int, int], list[str]] = {}
    for raw_pattern in patterns:
        label = raw_pattern.strip()
        needle = label.casefold()
        if not needle or label in pattern_totals:
            continue
        seen: set[tuple[int, int]] = set()
        for index, folded_line in enumerate(folded_lines):
            if needle not in folded_line:
                continue
            window = (
                max(1, index + 1 - context_lines),
                min(len(lines), index + 1 + context_lines),
            )
            seen.add(window)
            tags_by_window.setdefault(window, []).append(label)
        pattern_totals[label] = len(seen)

    ordered_windows = sorted(tags_by_window)
    line_budget = min(max_line_chars, max_chars)
    used_chars = 0
    attempted: set[int] = set()
    covered: set[int] = set()
    attempted_end = 0
    segment_results: list[GrepSegmentResult] = []
    for offset, requested_count in segments:
        cap = min(requested_count, max_matches_per_segment)
        segment_end = min(offset + cap, len(ordered_windows))
        attempted.update(range(offset, segment_end))
        attempted_end = max(attempted_end, segment_end)
        matches: list[DocumentGrepMatch] = []
        for position in range(offset, segment_end):
            window = ordered_windows[position]
            rendered = _render_grep_window(
                lines,
                window,
                used_chars=used_chars,
                max_chars=max_chars,
                line_budget=line_budget,
            )
            if rendered is None:
                break
            rendered.hit_patterns = tags_by_window[window]
            matches.append(rendered)
            used_chars += len(rendered.content)
            covered.add(position)
        segment_results.append(
            GrepSegmentResult(
                offset=offset,
                requested=requested_count,
                returned=len(matches),
                matches=matches,
            )
        )
    # has_more 只报系统截断：预算没装下的被尝试窗口，或流尾超出本次尝试范围。
    # 模型自选跳过的区间不算截断——它们不是"下一页"，是要就再发一段的显式选择。
    has_more = bool(attempted - covered) or attempted_end < len(ordered_windows)
    return GrepPatternsResult(
        pattern_totals=pattern_totals,
        total_windows=len(ordered_windows),
        segments=segment_results,
        has_more=has_more,
    )
