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
    """单个查询词一次 grep 的结果：命中窗口，外加说明截断程度与续读方式的三个字段。

    截断、翻页与整窗不拆行的完整行为契约见实现所在 materials.store.grep_lines；
    本处只描述字段级语义。
    """

    query: str
    matches: list[DocumentGrepMatch] = Field(default_factory=list)
    total_matches: int = Field(ge=0)  # 截断之前，去重后该词在全文的命中窗口总数
    has_more: bool  # 受命中数上限或字符预算影响，仍有未返回的后续窗口
    next_offset: int = Field(ge=0)  # 续读后续窗口时传入的 offset；无后续窗口时为 0
