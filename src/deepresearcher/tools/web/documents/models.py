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


class GrepResult(BaseModel):
    """单个查询词一次 grep 的结果:窗口 + 可续读的截断账。

    一次调用只查一个词——批量由 agent 在同一回合并行发起多个 GrepDocument 实现
    (该工具非 serial),因此不存在跨查询词的配额抢占/饿死,也无需 per-query 矩阵。
    截断(受 max_matches / max_chars 约束)通过 `total_matches / has_more / next_offset`
    显式回报:模型据此带 offset 翻页取回后续,或直接对返回的 start_line/end_line
    调 ReadDocument 展开。把 ripgrep `--count` 与 Claude Code `head_limit`+`offset`
    的思路落到 agent 工具上,而不是把内容腰斩。
    """

    query: str
    matches: list[DocumentGrepMatch] = Field(default_factory=list)
    total_matches: int = Field(ge=0)  # 去重后该词全文命中数(截断前)
    has_more: bool  # 仍有未取的后续窗口(上限或字符预算所致)
    next_offset: int = Field(ge=0)  # 取后续窗口的 offset;无后续则 0
