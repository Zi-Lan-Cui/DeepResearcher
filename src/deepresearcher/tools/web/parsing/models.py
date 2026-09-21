"""解析产出与文档结构块的共享 schema。"""

from enum import Enum
from typing import Literal, NotRequired, TypedDict


class DocumentModality(str, Enum):
    TEXT = "text"
    RICH = "rich"
    BINARY = "binary"


class DocumentBlock(TypedDict):
    block_id: str
    block_type: Literal["heading", "paragraph", "list_item", "table", "quote", "code"]
    text: str
    heading_path: NotRequired[list[str]]
    order: NotRequired[int]
    char_start: NotRequired[int]
    char_end: NotRequired[int]


class ParsedContent(TypedDict):
    """解析产出的字段基座(text/title/blocks);SourceDocument 在其上叠加抓取与缓存状态。

    解析函数按约定返回元组——plain 版给 (title, text),*_blocks 版给
    (title, text, blocks);装配进本 schema 的责任在编排方(Provider/FetchService)。
    """

    text: str
    title: NotRequired[str]
    blocks: NotRequired[list[DocumentBlock]]
