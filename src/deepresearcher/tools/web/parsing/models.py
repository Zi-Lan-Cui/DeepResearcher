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
    """格式解析器的纯输出，不包含网络抓取与缓存状态。"""

    text: str
    title: NotRequired[str]
    blocks: NotRequired[list[DocumentBlock]]
