"""测试共享 fake 与类型收口助手。

生产构造器的 ``llm`` / ``state`` 参数是具体类型（LLMInvoker / ResearchState）；
结构化兼容的测试替身经由 ``as_llm`` / ``as_state`` 单点收口，
把 ``cast`` 集中在这一处，而不是散落到每个调用点。
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Any, cast

from langchain_core.messages import AIMessage, ToolMessage

from deepresearcher.agents.researcher import ResearchAgent
from deepresearcher.agents.writer import CompleteReport, ReadEvidence, ReportWriter
from deepresearcher.config import AgentConfig
from deepresearcher.evidence.models import Evidence
from deepresearcher.llm import LLMInvoker
from deepresearcher.schemas import (
    Citation,
    MarkdownReportDraft,
    ParagraphBinding,
    ReadWorkingSet,
    ReleaseEvidence,
    ReportBrief,
    ResearchDirectionComplete,
    ResearchDirectionDecision,
    RestoreEvidence,
    SearchSources,
    WriterDirective,
)
from deepresearcher.state import ResearchState, SubTask
from deepresearcher.tools import SearchTool


def as_llm(fake: Any) -> LLMInvoker:
    """把结构化兼容的 fake LLM 收口为 LLMInvoker 类型。"""
    return cast(LLMInvoker, fake)


def as_state(partial: dict[str, Any]) -> ResearchState:
    """把部分 State 字典收口为 ResearchState（TypedDict 仅用于注释键）。"""
    return cast(ResearchState, partial)


def read_events(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 事件流中的全部 event 记录。"""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def event_types(path: Path) -> list[str]:
    return [item["event_type"] for item in read_events(path)]


def evidence(evidence_id: str, *, task_id: str) -> Evidence:
    """构造 Supervisor/Researcher 测试使用的最小有效 Evidence。"""
    return Evidence(
        evidence_id=f"{evidence_id}-ev",
        subtask_id=task_id,
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
    support="direct",
    direction="",
    retrieval="origin_fetch",
):
    """构造合法 Evidence 模型；State 通道已类型化，测试 fixture 不再用裸 dict。"""
    return Evidence(
        evidence_id=evidence_id,
        subtask_id="r1-1",
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
    "covered_topics": [
        {"topic": "核心结论", "role": "主线", "reason": "直接回答问题", "required": True}
    ],
    "required_points": ["给出有来源的结论"],
    "caveats": [],
}


class TextLLM:
    """即时回答假模型：只有文本接口与 bind_tools。"""

    async def ainvoke_text(self, _messages, **_kwargs):
        return type("Response", (), {"content": "这是显式注入的即时回答。"})()

    def bind_tools(self, _tools, **_kwargs):
        return self


class GraphLLM:
    """只测试图装配和路由时使用的最小工具绑定假模型。"""

    def bind_tools(self, _tools, tool_choice="any", **_kwargs):
        return self


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
        if latest_tool.name == CompleteReport.__name__ and "已通过本地引用校验" in str(
            latest_tool.content
        ):
            return AIMessage(content="报告已提交。")
        draft = await self.draft_factory(object(), MarkdownReportDraft, messages)
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": CompleteReport.__name__,
                    "args": draft.model_dump(),
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


class DirectionLLM:
    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.seen_messages: list[list] = []

    def bind_tools(self, _tools, **_kwargs):
        return self

    async def ainvoke(self, _messages):
        self.seen_messages.append(list(_messages))
        if any(
            isinstance(message, ToolMessage) and message.name == "ResearchDirectionComplete"
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
            args = ResearchDirectionComplete(
                reason=decision.reason,
                selected_evidence_ids=evidence_ids,
                conclusion=decision.conclusion,
                remaining_gaps=decision.remaining_gaps,
            ).model_dump()
            name = "ResearchDirectionComplete"
        return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "call"}])

    async def ainvoke_structured(self, schema, messages, **_kwargs):
        assert schema is ResearchDirectionDecision
        self.seen_messages.append(list(messages))
        return next(self.decisions)


class FakeSearchClient:
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
                        subtask_id=_task["id"],
                        research_direction=_task["question"],
                        claim=f"{candidate['title']} 的直接事实",
                        quote="可验证原文",
                        source_url=candidate["url"],
                        support="direct",
                    )
                ],
            }
        finally:
            self.active -= 1


def writer_llm(draft_factory):
    return _WriterToolLLM(draft_factory)


def write_with(draft_output, state, *, event_sink=None):
    """用 Writer 工具协议运行测试；不绕过 ReadEvidence/CompleteReport。"""
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
        ReportWriter(
            writer_llm(draft_output),
            AgentConfig(),
            render_incomplete=lambda _state: "incomplete",
            event_sink=event_sink,
        ).run(state)
    )


def researcher_agent(config, decisions, reader=None):
    return ResearchAgent(
        DirectionLLM(decisions),
        config,
        search_tool=SearchTool(FakeSearchClient()),
        reader_tool=reader or FakeReader(),
    )


TASK: SubTask = {
    "id": "r1-1",
    "question": "验证一个具体研究方向",
    "type": "search",
    "status": "pending",
    "assigned_agent": "research_agent",
}
