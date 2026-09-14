import re

from bs4 import BeautifulSoup

from deepresearcher.tools.web.parsing.models import DocumentBlock


def parse_html(data: bytes) -> tuple[str, str]:
    title, text, _ = parse_html_blocks(data)
    return title, text


def parse_html_blocks(data: bytes) -> tuple[str, str, list[DocumentBlock]]:
    soup = BeautifulSoup(data, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    for element in soup(
        ["script", "style", "noscript", "nav", "footer", "header", "aside", "form"]
    ):
        element.decompose()
    for element in soup.select('[role="complementary"], [role="navigation"]'):
        element.decompose()

    # 优先解析语义化主体。页面同时存在多个候选时取正文文本最长者；
    # 找不到 article/main 才回退整个文档，兼容旧页面与简单 HTML。
    candidates = soup.find_all(["article", "main"])
    root = (
        max(candidates, key=lambda item: len(item.get_text(" ", strip=True)))
        if candidates
        else soup
    )
    blocks: list[DocumentBlock] = []
    heading_path: list[str] = []
    for element in root.find_all(
        ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote", "pre", "table"]
    ):
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
