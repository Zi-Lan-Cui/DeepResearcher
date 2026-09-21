"""持久文档及其有界读取结果。"""

from pydantic import BaseModel, Field

from deepresearcher.vocab import RETRIEVAL_ORIGIN_FETCH


class DocumentOutlineItem(BaseModel):
    heading: str
    line: int = Field(ge=1)


class DocumentRef(BaseModel):
    """ResearchMaterialStore 中正文的稳定句柄；不携带完整正文。"""

    document_id: str
    title: str = ""
    source_url: str
    published_at: str = ""
    retrieval_method: str = RETRIEVAL_ORIGIN_FETCH
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


class GrepResult(BaseModel):
    """单个查询词一次 grep 的结果:窗口 + 可续读的截断账。

    截断/翻页/整窗不截半行的完整行为契约见 materials.store.grep_lines(实现所在);
    本处只描述字段级语义。
    """

    query: str
    matches: list[DocumentGrepMatch] = Field(default_factory=list)
    total_matches: int = Field(ge=0)  # 去重后该词全文命中数(截断前)
    has_more: bool  # 仍有未取的后续窗口(上限或字符预算所致)
    next_offset: int = Field(ge=0)  # 取后续窗口的 offset;无后续则 0
