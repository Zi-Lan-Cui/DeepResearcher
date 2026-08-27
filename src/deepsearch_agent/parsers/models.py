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


class ParsedDocument(TypedDict):
    text: str
    status: NotRequired[Literal["completed", "failed"]]
    source_url: NotRequired[str]
    final_url: NotRequired[str]
    name: NotRequired[str]
    ext: NotRequired[str]
    content_type: NotRequired[str]
    modality: NotRequired[str]
    title: NotRequired[str]
    blocks: NotRequired[list[DocumentBlock]]
    raw_bytes: NotRequired[int]
    status_code: NotRequired[int]
    content_hash: NotRequired[str]
    error: NotRequired[str]
    error_code: NotRequired[str]
    retrieval_method: NotRequired[str]
    support_ceiling: NotRequired[str]
    fetch_duration_ms: NotRequired[float]
    parse_duration_ms: NotRequired[float]
