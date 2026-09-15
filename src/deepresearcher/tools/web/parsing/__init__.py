"""网页与文档格式解析；只解析，不下载、不切 chunk。"""

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
