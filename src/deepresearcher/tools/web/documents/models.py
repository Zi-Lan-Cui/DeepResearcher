"""持久文档及其有界读取结果。"""

from pydantic import BaseModel, Field


class DocumentOutlineItem(BaseModel):
    heading: str
    line: int = Field(ge=1)


class DocumentRef(BaseModel):
    """ResearchMaterialStore 中正文的稳定句柄；不携带完整正文。"""

    document_id: str
    title: str = ""
    source_url: str
    published_at: str = ""
    retrieval_method: str = "origin_fetch"
    support_ceiling: str = "direct"
    content_hash: str
    line_count: int = Field(ge=0)
    token_count: int = Field(ge=0)
    outline: list[DocumentOutlineItem] = Field(default_factory=list)


class DocumentView(DocumentRef):
    """ReadSources 返回给 Researcher 的来源视图。"""

    inline: bool = False
    content: str = ""


class DocumentReadRange(BaseModel):
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    content: str


class DocumentGrepMatch(BaseModel):
    query: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    content: str
