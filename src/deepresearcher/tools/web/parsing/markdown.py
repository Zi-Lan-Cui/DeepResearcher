"""将 Markdown 归一化为可定位的文档块。"""

import re
from typing import Literal

from deepresearcher.tools.web.parsing.models import DocumentBlock

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
_LIST_ITEM = re.compile(r"^\s*(?:[-+*]|\d+[.)])\s+(.+)$")


def parse_markdown(data: bytes) -> tuple[str, str, list[DocumentBlock]]:
    source = data.decode("utf-8", errors="replace").strip()
    blocks: list[DocumentBlock] = []
    headings: list[str] = []
    title = ""
    paragraph: list[str] = []

    def append(
        block_type: Literal["heading", "paragraph", "list_item", "table", "quote", "code"],
        value: str,
    ) -> None:
        text = " ".join(value.split()).strip()
        if not text:
            return
        blocks.append(
            {
                "block_id": f"b-{len(blocks):04d}",
                "block_type": block_type,
                "text": text,
                "heading_path": list(headings),
                "order": len(blocks),
            }
        )

    def flush_paragraph() -> None:
        if paragraph:
            append("paragraph", " ".join(paragraph))
            paragraph.clear()

    in_code = False
    code_lines: list[str] = []
    for raw_line in source.splitlines():
        line = raw_line.rstrip()
        if line.lstrip().startswith("```"):
            if in_code:
                append("code", "\n".join(code_lines))
                code_lines.clear()
            else:
                flush_paragraph()
            in_code = not in_code
            continue
        if in_code:
            code_lines.append(line)
            continue
        heading = _HEADING.match(line)
        if heading:
            flush_paragraph()
            level, value = len(heading.group(1)), heading.group(2).strip()
            headings[level - 1 :] = [value]
            if not title:
                title = value
            append("heading", value)
            continue
        item = _LIST_ITEM.match(line)
        if item:
            flush_paragraph()
            append("list_item", item.group(1))
            continue
        if line.startswith(">"):
            flush_paragraph()
            append("quote", line.lstrip("> "))
            continue
        if not line.strip():
            flush_paragraph()
            continue
        paragraph.append(line)
    flush_paragraph()
    if code_lines:
        append("code", "\n".join(code_lines))
    return title, "\n\n".join(block["text"] for block in blocks), blocks
