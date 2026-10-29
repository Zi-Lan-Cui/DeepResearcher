"""测试共享替身与类型转换助手：生产构造器要具体类型，替身经 as_llm / as_state 单点转换。"""

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Literal, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel, Field, model_validator

from deepresearcher.agents.researcher import Researcher
from deepresearcher.agents.writer import ReadEvidence, ReportComplete, Writer
from deepresearcher.config import AgentConfig
from deepresearcher.evidence.models import Evidence
from deepresearcher.schemas import (
    Citation,
    MarkdownReportDraft,
    ParagraphBinding,
    ReadWorkingSet,
    ReleaseEvidence,
    ReportBrief,
    RestoreEvidence,
    SearchSources,
    Task,
    TaskComplete,
    WriterDirective,
)
from deepresearcher.schemas.limits import (
    EVIDENCE_REFERENCES_HARD_LIMIT,
    SEARCH_QUERIES_PER_CALL,
    SOURCE_CANDIDATES_PER_READ,
)
from deepresearcher.state import RunState
from deepresearcher.tools import SearchTool
from deepresearcher.tools.web.documents import (
    DocumentOutlineItem,
    DocumentReadRange,
    DocumentRef,
    GrepPatternsResult,
)
from deepresearcher.tools.web.materials.models import (
    SearchResultSet,
    StoredDocument,
    build_stored_document,
)
from deepresearcher.tools.web.materials.paging import grep_patterns, read_lines
from deepresearcher.vocab import RETRIEVAL_ORIGIN_FETCH


def as_llm(fake: Any) -> BaseChatModel:
    """把结构化兼容的 fake LLM 转换为 BaseChatModel 类型。"""
    return cast(BaseChatModel, fake)


def as_state(partial: dict[str, Any]) -> RunState:
    """把部分 State 字典转换为 RunState。TypedDict 只在类型标注层生效，不做运行时校验。"""
    return cast(RunState, partial)


def read_events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def event_types(path: Path) -> list[str]:
    return [item["event_type"] for item in read_events(path)]


def evidence(evidence_id: str, *, task_id: str) -> Evidence:
    """构造 Supervisor/Researcher 测试使用的最小有效 Evidence。"""
    return Evidence(
        evidence_id=f"{evidence_id}-ev",
        task_id=task_id,
        research_direction="测试方向",
        claim=f"{evidence_id} 的直接事实",
        quote="可验证原文。",
        source_url=f"https://example.com/{evidence_id}",
    )


def make_evidence(
    evidence_id,
    claim,
    *,
    quote=None,
    url="https://example.com/a",
    support="strict",
    direction="",
    retrieval="origin_fetch",
):
    """构造合法 Evidence 模型。"""
    return Evidence(
        evidence_id=evidence_id,
        task_id="r1-1",
        research_direction=direction,
        claim=claim,
        quote=quote if quote is not None else claim,
        source_url=url,
        support=support,
        retrieval_method=retrieval,
    )


def make_binding(text, evidence_ids=None, kind=None):
    if kind is None:
        kind = "evidence" if evidence_ids else "transition"
    return ParagraphBinding(text=text, kind=kind, evidence_ids=evidence_ids or [])


def make_citation(source_id, claim="事实", quote="原文", url=""):
    return Citation(id=source_id, claim=claim, quote=quote, url=url)


REPORT_BRIEF = {
    "answer_goal": "回答测试问题",
    "chapters": [
        {"topic": "核心结论", "role": "主线", "conclusion": "直接回答问题", "required": True}
    ],
    "caveats": [],
}


class _WriterToolRunnable:
    def __init__(self, draft_factory):
        self.draft_factory = draft_factory

    async def ainvoke(self, messages):
        tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
        if not tool_messages:
            ids = re.findall(r"evidence_id=([^ |\\n]+)", str(messages[-1].content))
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": ReadEvidence.__name__,
                        "args": {"evidence_ids": ids, "reason": "测试读取目录中的证据"},
                        "id": "read-1",
                    }
                ],
            )
        latest_tool = tool_messages[-1]
        if latest_tool.name == ReportComplete.__name__ and "已通过本地引用校验" in str(
            latest_tool.content
        ):
            return AIMessage(content="报告已提交。")
        draft = await self.draft_factory(object(), MarkdownReportDraft, messages)
        payload = draft.model_dump()
        payload.setdefault("title", "测试报告标题")
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": ReportComplete.__name__,
                    "args": payload,
                    "id": "complete-1",
                }
            ],
        )


class _WriterToolLLM:
    def __init__(self, draft_factory):
        self.draft_factory = draft_factory

    def bind_tools(self, _tools, **_kwargs):
        return _WriterToolRunnable(self.draft_factory)


class _WriterNoToolRunnable:
    async def ainvoke(self, messages):
        del messages
        return AIMessage(content="这是没有提交的普通文本。")


class _WriterNoToolLLM:
    def bind_tools(self, _tools, **_kwargs):
        return _WriterNoToolRunnable()


class ResearcherDecision(BaseModel):
    """Researcher 测试脚本的一步意图，由 ResearcherLLM 翻译成真实工具调用。

    互斥规则只服务于测试构造，生产端靠类型分离保证。
    """

    action: Literal["search", "read", "inspect", "release", "restore", "complete"]
    reason: str
    queries: list[str] = Field(default_factory=list, max_length=SEARCH_QUERIES_PER_CALL)
    candidate_ids: list[str] = Field(default_factory=list, max_length=SOURCE_CANDIDATES_PER_READ)
    evidence_ids: list[str] = Field(default_factory=list, max_length=EVIDENCE_REFERENCES_HARD_LIMIT)
    conclusion: str = ""

    @model_validator(mode="after")
    def validate_action(self) -> "ResearcherDecision":
        if self.action == "search" and (not self.queries or self.candidate_ids):
            raise ValueError("action=search 时必须提供至少一条查询。")
        if self.action == "read" and (not self.candidate_ids or self.queries):
            raise ValueError("action=read 时必须提供候选 ID，且不得继续提供查询。")
        if self.action == "inspect" and (self.queries or self.candidate_ids or self.evidence_ids):
            raise ValueError("action=inspect 时不得提供查询、候选 ID 或 Evidence ID。")
        if self.action in {"release", "restore"} and (
            not self.evidence_ids or self.queries or self.candidate_ids
        ):
            raise ValueError(
                f"action={self.action} 时必须提供 Evidence ID，且不得提供查询或候选 ID。"
            )
        if self.action == "complete" and (self.queries or self.candidate_ids or self.evidence_ids):
            raise ValueError("action=complete 时不得继续提供查询、候选 ID 或 Evidence ID。")
        if self.action != "complete" and self.conclusion.strip():
            raise ValueError("只有 action=complete 时才能输出 conclusion。")
        return self


class ResearcherLLM:
    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.seen_messages: list[list] = []

    def bind_tools(self, _tools, **_kwargs):
        return self

    async def ainvoke(self, _messages):
        self.seen_messages.append(list(_messages))
        if any(
            isinstance(message, ToolMessage) and message.name == "TaskComplete"
            for message in _messages
        ):
            return AIMessage(content="")
        decision = next(self.decisions)
        if decision.action == "search":
            args = SearchSources(
                reason=decision.reason,
                queries=decision.queries,
            ).model_dump()
            name = "SearchSources"
        elif decision.action == "read":
            args = {
                "reason": decision.reason,
                "candidate_ids": decision.candidate_ids,
            }
            name = "ReadSources"
        elif decision.action == "inspect":
            args = ReadWorkingSet(reason=decision.reason).model_dump()
            name = "ReadWorkingSet"
        elif decision.action == "release":
            args = ReleaseEvidence(
                evidence_ids=decision.evidence_ids,
                reason=decision.reason,
            ).model_dump()
            name = "ReleaseEvidence"
        elif decision.action == "restore":
            args = RestoreEvidence(
                evidence_ids=decision.evidence_ids,
                reason=decision.reason,
            ).model_dump()
            name = "RestoreEvidence"
        else:
            evidence_ids = list(
                dict.fromkeys(
                    re.findall(
                        r'"evidence_id"\s*:\s*"([^"]+)"',
                        "\n".join(str(message.content) for message in _messages),
                    )
                )
            )
            args = TaskComplete(
                reason=decision.reason,
                selected_evidence_ids=evidence_ids,
                conclusion=decision.conclusion,
            ).model_dump()
            name = "TaskComplete"
        return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "call"}])


class FakeSearchService:
    async def asearch(self, query):
        return [
            {"title": query, "url": f"https://example.com/{query}/a", "snippet": "", "score": 0.9},
            {"title": query, "url": f"https://example.com/{query}/b", "snippet": "", "score": 0.8},
        ]


class FakeReader:
    def __init__(self):
        self.active = 0
        self.max_active = 0

    async def arun(self, _task, candidate):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.01)
            return {
                "task_id": _task["id"],
                "status": "completed",
                "source_url": candidate["url"],
                "evidences": [
                    Evidence(
                        evidence_id=f"ev-{len(candidate['url'])}",
                        task_id=_task["id"],
                        research_direction=_task["question"],
                        claim=f"{candidate['title']} 的直接事实",
                        quote="可验证原文",
                        source_url=candidate["url"],
                        support="strict",
                    )
                ],
            }
        finally:
            self.active -= 1


class MemoryResearchMaterialStore:
    """进程内材料存储：测试专用，免外部 Redis 依赖。"""

    def __init__(self) -> None:
        self._searches: dict[tuple[str, str], SearchResultSet] = {}
        self._documents: dict[str, StoredDocument] = {}
        self._fetch_index: dict[str, str] = {}

    async def put_search_results(self, result_set: SearchResultSet) -> None:
        self._searches[(result_set.run_id, result_set.search_id)] = result_set.model_copy(deep=True)

    async def get_search_results(self, run_id: str, search_id: str) -> SearchResultSet:
        try:
            return self._searches[(run_id, search_id)].model_copy(deep=True)
        except KeyError as exc:
            raise FileNotFoundError(f"搜索结果不存在：{search_id}") from exc

    async def put(
        self,
        *,
        text: str,
        title: str,
        source_url: str,
        published_at: str = "",
        retrieval_method: str = RETRIEVAL_ORIGIN_FETCH,
        support_ceiling: str = "strict",
        token_count: int = 0,
        outline: list[DocumentOutlineItem] | None = None,
        fetch_key: str = "",
    ) -> DocumentRef:
        document = build_stored_document(
            text=text,
            title=title,
            source_url=source_url,
            published_at=published_at,
            retrieval_method=retrieval_method,
            support_ceiling=support_ceiling,
            token_count=token_count,
            outline=outline or [],
        )
        ref = document.ref
        self._documents[ref.document_id] = document
        if fetch_key:
            self._fetch_index[fetch_key] = ref.document_id
        return ref

    async def resolve_fetch(self, fetch_key: str) -> DocumentRef | None:
        document_id = self._fetch_index.get(fetch_key)
        document = self._documents.get(document_id or "")
        return document.ref.model_copy(deep=True) if document is not None else None

    async def get(self, document_id: str) -> DocumentRef:
        return (await self._document(document_id)).ref.model_copy(deep=True)

    async def text(self, document_id: str) -> str:
        return (await self._document(document_id)).content

    async def read(
        self,
        document_id: str,
        ranges: list[tuple[int, int]],
        *,
        max_lines: int,
        max_chars: int,
    ) -> list[DocumentReadRange]:
        document = await self._document(document_id)
        return read_lines(
            document.content.splitlines(),
            ranges,
            max_lines=max_lines,
            max_chars=max_chars,
        )

    async def grep_patterns(
        self,
        document_id: str,
        patterns: list[str],
        *,
        context_lines: int,
        segments: list[tuple[int, int]],
        max_matches_per_segment: int,
        max_chars: int,
        max_line_chars: int,
    ) -> GrepPatternsResult:
        document = await self._document(document_id)
        return grep_patterns(
            document.content.splitlines(),
            patterns,
            context_lines=context_lines,
            segments=segments,
            max_matches_per_segment=max_matches_per_segment,
            max_chars=max_chars,
            max_line_chars=max_line_chars,
        )

    async def close(self) -> None:
        return None

    async def _document(self, document_id: str) -> StoredDocument:
        try:
            return self._documents[document_id]
        except KeyError as exc:
            raise FileNotFoundError(f"文档不存在：{document_id}") from exc


def writer_llm(draft_factory):
    return _WriterToolLLM(draft_factory)


def write_with(draft_output, state, *, event_sink=None):
    """用 Writer 工具协议运行测试；不绕过 ReadEvidence/ReportComplete。"""
    state = dict(state)
    state.setdefault(
        "writer_directive",
        WriterDirective(
            query=state.get("clarified_query", "测试问题"),
            report_brief=ReportBrief.model_validate(REPORT_BRIEF),
            research_status="completed",
            generation_mode="full",
        ),
    )
    return asyncio.run(
        Writer(
            writer_llm(draft_output),
            AgentConfig(),
            render_incomplete=lambda _state: "incomplete",
            event_sink=event_sink,
        ).run(state)
    )


def researcher_agent(config, decisions, reader=None):
    return Researcher(
        ResearcherLLM(decisions),
        config,
        search_tool=SearchTool(FakeSearchService()),
        reader_tool=reader or FakeReader(),
    )


TASK: Task = {
    "id": "r1-1",
    "question": "验证一个具体研究方向",
}
