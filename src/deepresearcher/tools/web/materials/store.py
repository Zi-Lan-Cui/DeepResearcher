"""研究材料存储协议与纯函数读取逻辑。"""

from typing import Protocol

from deepresearcher.tools.web.documents import (
    DocumentGrepMatch,
    DocumentOutlineItem,
    DocumentReadRange,
    DocumentRef,
    GrepResult,
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

    async def grep(
        self,
        document_id: str,
        query: str,
        *,
        context_lines: int,
        max_matches: int,
        max_chars: int,
        max_line_chars: int,
        offset: int = 0,
    ) -> GrepResult: ...

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


def grep_lines(
    lines: list[str],
    query: str,
    *,
    context_lines: int,
    max_matches: int,
    max_chars: int,
    max_line_chars: int,
    offset: int = 0,
) -> GrepResult:
    """对已加载文本执行单查询词的字面检索，返回截断可见、可续读的窗口结果。

    一次调用只查一个词；批量查多个词由 agent 在同一回合并行发起多个
    GrepDocument 实现，该工具非 serial，因此不存在跨查询词相互占用窗口的情况，
    offset 的含义就是跳过该词去重后的前 N 个窗口。

    预算分三层上限：整页字符上限 max_chars、每页窗口数上限 max_matches、单行字符
    上限 max_line_chars。超长行裁切并标注剩余字符数；窗口因页预算被裁尾时，
    裁掉的行数记在 tail_omitted_lines 并在正文容得下时补一行标注——
    被裁的尾部不属于下一页，offset 按窗口整体推进，补读走 ReadDocument 按行号。
    单行先裁切再计入预算，每页至少发出一个窗口首行，翻页必定前进。
    """

    needle = query.strip().casefold()
    # 全文扫描得到去重后的候选窗口，按文档行号排序；total_matches 取截断前的该列表总数。
    windows: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    if needle:
        for index, line in enumerate(lines):
            if needle not in line.casefold():
                continue
            window = (
                max(1, index + 1 - context_lines),
                min(len(lines), index + 1 + context_lines),
            )
            if window in seen:
                continue
            seen.add(window)
            windows.append(window)

    line_budget = min(max_line_chars, max_chars)
    matches: list[DocumentGrepMatch] = []
    used_chars = 0
    for start, end in windows[offset : offset + max_matches]:
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
            break
        emitted_line_count = len(rendered)
        tail_omitted_lines = end - cut_line + 1 if cut_line is not None else 0
        if tail_omitted_lines:
            note = _GREP_WINDOW_OMITTED_TEMPLATE.format(
                lines=tail_omitted_lines, start=cut_line, end=end
            )
            if used_chars + window_chars + 1 + len(note) <= max_chars:
                rendered.append(note)
                window_chars += 1 + len(note)
        content = "\n".join(rendered)
        matches.append(
            DocumentGrepMatch(
                query=query,
                start_line=start,
                end_line=start + emitted_line_count - 1,
                content=content,
                tail_omitted_lines=tail_omitted_lines,
            )
        )
        used_chars += len(content)

    consumed = offset + len(matches)
    has_more = consumed < len(windows)
    return GrepResult(
        query=query,
        matches=matches,
        total_matches=len(windows),
        has_more=has_more,
        next_offset=consumed if has_more else 0,
    )
