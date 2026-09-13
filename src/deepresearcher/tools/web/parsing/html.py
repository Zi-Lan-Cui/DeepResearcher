import re

from bs4 import BeautifulSoup

from deepresearcher.tools.web.parsing.models import DocumentBlock


def parse_html(data: bytes) -> tuple[str, str]:
    title, text, _ = parse_html_blocks(data)
    return title, text


def parse_html_blocks(data: bytes) -> tuple[str, str, list[DocumentBlock]]:
    soup = BeautifulSoup(data, "html.parser")
    for element in soup(["script", "style", "noscript", "nav", "footer", "header"]):
        element.decompose()
    title = soup.title.get_text(strip=True) if soup.title else ""
    blocks: list[DocumentBlock] = []
    heading_path: list[str] = []
    for element in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "blockquote", "pre", "table"]):
        raw = element.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", raw).strip()
        if not text:
            continue
        if element.name and element.name.startswith("h"):
            level = int(element.name[1:])
            heading_path = heading_path[: level - 1] + [text]
            block_type = "heading"
        elif element.name == "li":
            block_type = "list_item"
        elif element.name == "blockquote":
            block_type = "quote"
        elif element.name == "pre":
            block_type = "code"
        elif element.name == "table":
            block_type = "table"
        else:
            block_type = "paragraph"
        blocks.append(
            {
                "block_id": f"b-{len(blocks):04d}",
                "block_type": block_type,
                "text": text,
                "heading_path": list(heading_path),
                "order": len(blocks),
            }
        )
    text = "\n".join(block["text"] for block in blocks)
    return title, text, blocks
