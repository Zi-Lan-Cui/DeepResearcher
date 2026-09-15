from deepresearcher.tools.web.parsing.models import DocumentBlock


def parse_text(data: bytes) -> tuple[str, str]:
    return "", data.decode("utf-8", errors="replace").strip()


def parse_text_blocks(data: bytes) -> tuple[str, str, list[DocumentBlock]]:
    """将普通文本按非空行转为可定位块。"""
    title, text = parse_text(data)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    blocks: list[DocumentBlock] = [
        {
            "block_id": f"b-{index:04d}",
            "block_type": "paragraph",
            "text": line,
            "heading_path": [],
            "order": index,
        }
        for index, line in enumerate(lines)
    ]
    return title, text, blocks
