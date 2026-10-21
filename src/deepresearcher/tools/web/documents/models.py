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
    hit_patterns: list[str]  # 命中该窗口的词；跨词同窗时并存
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    content: str
    # 窗口尾部因页字符预算未发出的行数；0 表示窗口完整。补读走 ReadDocument 按行号。
    tail_omitted_lines: int = Field(default=0, ge=0)


class GrepSegmentResult(BaseModel):
    offset: int = Field(ge=0)
    requested: int = Field(ge=1)
    returned: int = Field(ge=0)
    matches: list[DocumentGrepMatch] = Field(default_factory=list)


class GrepPatternsResult(BaseModel):
    """多关键词一次检索的归并结果：offset 恒指归并命中流中的窗口位置。

    截断、翻页与行级裁切的完整行为契约见实现所在 materials.store.grep_patterns；
    本处只描述字段级语义。
    """

    pattern_totals: dict[str, int]  # 每词单独去重后的命中窗口总数
    total_windows: int = Field(ge=0)  # 跨词去重合并后的流长度
    segments: list[GrepSegmentResult] = Field(default_factory=list)
    has_more: bool  # 预算未容下的被试窗口仍存在，或流尾超出本次尝试范围；模型自选跳过的段不算
