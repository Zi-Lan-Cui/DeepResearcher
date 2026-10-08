"""网页与文档格式解析；只解析，不下载、不切 chunk。

返回形状按名字规律：plain 版如 parse_html、parse_pdf 返回 (title, text)，
*_blocks 后缀版返回 (title, text, blocks)。pdf、docx 只有 plain 版：
版式文档不产生结构化块，blocks 不出现是契约规定，不是实现缺口。
"""

from deepresearcher.tools.web.parsing.docx import parse_docx
from deepresearcher.tools.web.parsing.html import parse_html, parse_html_blocks
from deepresearcher.tools.web.parsing.markdown import parse_markdown
from deepresearcher.tools.web.parsing.models import DocumentBlock, DocumentModality, ParsedContent
from deepresearcher.tools.web.parsing.pdf import parse_pdf
from deepresearcher.tools.web.parsing.text import parse_text, parse_text_blocks

__all__ = [
    "DocumentBlock",
    "DocumentModality",
    "ParsedContent",
    "parse_docx",
    "parse_html",
    "parse_html_blocks",
    "parse_markdown",
    "parse_pdf",
    "parse_text",
    "parse_text_blocks",
]
