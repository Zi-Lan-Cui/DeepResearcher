"""完整文档解析器；只解析，不下载、不切 chunk。"""

from deepsearch_agent.parsers.docx import parse_docx
from deepsearch_agent.parsers.html import parse_html, parse_html_blocks
from deepsearch_agent.parsers.models import DocumentBlock, DocumentModality, ParsedDocument
from deepsearch_agent.parsers.pdf import parse_pdf
from deepsearch_agent.parsers.text import parse_text

__all__ = [
    "DocumentBlock",
    "DocumentModality",
    "ParsedDocument",
    "parse_docx",
    "parse_html",
    "parse_html_blocks",
    "parse_pdf",
    "parse_text",
]
