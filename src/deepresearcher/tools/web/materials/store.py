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
        retrieval_method: str = "origin_fetch",
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
        offset: int = 0,
    ) -> GrepResult: ...

    async def close(self) -> None: ...


def read_lines(
    lines: list[str],
    ranges: list[tuple[int, int]],
    *,
    max_lines: int,
    max_chars: int,
) -> list[DocumentReadRange]:
    """对已加载文本做有界行号读取；存储后端共用同一语义。"""

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
                item = item[:remaining_chars]
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
    offset: int = 0,
) -> GrepResult:
    """对已加载文本执行单查询词的字面检索，返回**可见截断、可续读**的窗口结果。

    一次调用只查一个词:批量由 agent 在同一回合并行发起多个 GrepDocument 实现
    (该工具非 serial)。因此不存在跨查询词的配额抢占/饿死,`offset` 就是"跳过该词
    去重后的前 N 个窗口"这一单纯分页。截断(受 max_matches / max_chars 约束)通过
    total_matches/has_more/next_offset 显式回报;max_chars 触顶时**整窗不截半行**,
    留待下一页。为保证翻页必定前进,当某次调用一个窗口都还没发出、而下一个窗口又
    超过字符预算时,仍整窗发出这一次(受 context_lines 上限约束,单个窗口有限)。
    """

    needle = query.strip().casefold()
    # 全文扫描得到去重后的候选窗口(文档序);据此得出 total_matches(截断前)。
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

    matches: list[DocumentGrepMatch] = []
    used_chars = 0
    for start, end in windows[offset : offset + max_matches]:
        content = "\n".join(
            f"L{line_number}: {lines[line_number - 1]}" for line_number in range(start, end + 1)
        )
        # 整窗预算;仅在"已发出至少一个窗口"时因字符预算停手,保证至少前进一格、不死循环。
        if matches and used_chars + len(content) > max_chars:
            break
        matches.append(
            DocumentGrepMatch(query=query, start_line=start, end_line=end, content=content)
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
