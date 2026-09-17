"""研究材料存储协议与纯函数读取逻辑。"""

from typing import Protocol

from deepresearcher.tools.web.documents import (
    DocumentGrepMatch,
    DocumentOutlineItem,
    DocumentReadRange,
    DocumentRef,
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
        queries: list[str],
        *,
        context_lines: int,
        max_matches: int,
        max_chars: int,
    ) -> list[DocumentGrepMatch]: ...

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
    queries: list[str],
    *,
    context_lines: int,
    max_matches: int,
    max_chars: int,
) -> list[DocumentGrepMatch]:
    """对已加载文本执行字面检索，返回稳定行号窗口。"""

    results: list[DocumentGrepMatch] = []
    used_chars = 0
    seen_ranges: set[tuple[int, int, str]] = set()
    for query in dict.fromkeys(item.strip() for item in queries if item.strip()):
        needle = query.casefold()
        for index, line in enumerate(lines):
            if needle not in line.casefold():
                continue
            start = max(1, index + 1 - context_lines)
            end = min(len(lines), index + 1 + context_lines)
            key = (start, end, query)
            if key in seen_ranges:
                continue
            content = "\n".join(
                f"L{line_number}: {lines[line_number - 1]}" for line_number in range(start, end + 1)
            )
            remaining = max_chars - used_chars
            if remaining <= 0:
                return results
            content = content[:remaining]
            results.append(
                DocumentGrepMatch(
                    query=query,
                    start_line=start,
                    end_line=end,
                    content=content,
                )
            )
            seen_ranges.add(key)
            used_chars += len(content)
            if len(results) >= max_matches:
                return results
    return results
