"""LangGraph 顶层状态:跨节点交接物的类型与合并语义。

字段按所有权分组;channel 直接存 Pydantic 模型,节点不再在 dict 与
模型之间手工往返。合并 reducer 是防御性恢复的唯一入口:同一 run 内
LangGraph 保持模型对象,跨进程 checkpoint 恢复时 incoming 可能是 dict,
在 reducer 内恢复一次,消费节点拿到的永远是模型。
"""

from collections.abc import MutableMapping
from operator import add
from typing import Annotated, Literal, NotRequired, TypedDict, TypeVar, cast

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from deepresearcher.errors import AgentError
from deepresearcher.evidence.models import Evidence
from deepresearcher.observability.events.models import NodeEvent
from deepresearcher.routing import NodeName
from deepresearcher.schemas import (
    Citation,
    ParagraphBinding,
    ReportBrief,
    ResearchDirectionResult,
    ResearchProgress,
    ResearchSynthesis,
    ReviewProgress,
    RunLifecycle,
    WriterDirective,
    WriterProgress,
)


class StateInvariantError(AgentError):
    """节点产生了互相矛盾的运行状态。"""

    code = "invalid_state"


class SubTask(TypedDict):
    id: str
    run_id: NotRequired[str]
    question: str
    # Evidence 召回可选的内部细分；缺失时以 question 作为唯一检索查询。
    research_direction: NotRequired[str]
    subquestions: NotRequired[list[str]]
    round: NotRequired[int]
    sequence: NotRequired[int]
    type: Literal["search", "rag", "read", "memory"]
    status: Literal["pending", "failed"]
    assigned_agent: str
    worker_id: NotRequired[str]
    worker_index: NotRequired[int]
    parent_task_id: NotRequired[str]
    operation_id: NotRequired[str]


_ModelT = TypeVar("_ModelT", bound=BaseModel)


def _coerce(model_cls: type[_ModelT], value: object) -> _ModelT:
    """跨进程 checkpoint 恢复时值可能是 dict;恢复为模型,否则原样返回。"""
    if isinstance(value, model_cls):
        return value
    return model_cls.model_validate(value)


def merge_evidences(current: list[Evidence], incoming: list[Evidence]) -> list[Evidence]:
    """按 evidence_id 去重合并；同名但内容不同视为状态契约冲突。"""
    restored = [_coerce(Evidence, item) for item in [*current, *incoming]]
    by_id: dict[str, Evidence] = {}
    for item in restored:
        previous = by_id.get(item.evidence_id)
        if previous is not None and previous != item:
            raise StateInvariantError(f"Evidence ID 冲突且内容不同：{item.evidence_id}。")
        by_id[item.evidence_id] = item
    return list(by_id.values())


def merge_task_results(
    current: list[ResearchDirectionResult],
    incoming: list[ResearchDirectionResult],
) -> list[ResearchDirectionResult]:
    """按 task_id 去重合并;同一 task 重跑(恢复执行)时以新结果覆盖。"""
    restored = [_coerce(ResearchDirectionResult, item) for item in [*current, *incoming]]
    by_id: dict[str, ResearchDirectionResult] = {}
    for item in restored:
        by_id[item.task_id] = item
    return list(by_id.values())


def merge_unique(current: list[str], incoming: list[str]) -> list[str]:
    """字符串集合语义合并,保持首次出现顺序。"""
    return list(dict.fromkeys([*current, *incoming]))


class ResearchState(TypedDict, total=False):
    # State 只保存数据快照；Agent、LLM、工具和锁由 Graph 装配层持有。
    run: RunLifecycle
    research: ResearchProgress
    writer: WriterProgress
    review: ReviewProgress
    run_id: str

    # 输入与路由(Router / Clarify / Writer)
    query: str
    clarified_query: str
    session_id: str
    route: str
    route_reason: str
    answer_mode: Literal[
        "quick_answer",
        "deep_research",
        "research_incomplete",
        "review_limited",
        "clarification_needed",
    ]
    research_brief: str
    clarification_question: str
    draft_answer: str

    supervisor_messages: Annotated[list[BaseMessage], add]
    evidences: Annotated[list[Evidence], merge_evidences]
    # 已读取来源的快照引用（URL），不作为报告正文内容使用。
    source_refs: Annotated[list[str], merge_unique]
    # 当前研究 run 已尝试的规范化 URL；用于跨 Supervisor 回流维持去重，
    # 但不会共享到其他 session。
    attempted_source_urls: Annotated[list[str], merge_unique]
    # Supervisor 当前工作集；完整 Evidence 档案仍保存在 evidences 中。
    active_evidence_ids: list[str]
    working_set_revision: int
    research_synthesis: ResearchSynthesis | None
    report_brief: ReportBrief | None
    writer_directive: WriterDirective | None
    task_results: Annotated[list[ResearchDirectionResult], merge_task_results]
    supervisor_next: NodeName

    writer_draft: str
    # Writer 产出的 evidence_id 键草稿(含 [[cite:evidence_id]] 标记);
    # 审阅通过后由终检渲染层编号渲染为 report。
    report_draft: str
    paragraph_bindings: list[ParagraphBinding]
    citations: list[Citation]
    report: str

    # 审计(Instrumentation)
    evidence_count: int
    source_count: int
    node_events: Annotated[list[NodeEvent], add]


def restore_state_models(state: dict[str, object]) -> None:
    """恢复 JSON checkpoint 中被还原为 dict 的嵌套模型。"""
    scalar_models = (
        ("run", RunLifecycle),
        ("research", ResearchProgress),
        ("writer", WriterProgress),
        ("review", ReviewProgress),
        ("report_brief", ReportBrief),
        ("writer_directive", WriterDirective),
        ("research_synthesis", ResearchSynthesis),
    )
    for key, model_cls in scalar_models:
        value = state.get(key)
        if value is None:
            if key in {
                "report_brief",
                "writer_directive",
                "research_synthesis",
            }:
                continue
            state[key] = model_cls()  # type: ignore[call-arg]
        elif not isinstance(value, model_cls):
            state[key] = model_cls.model_validate(value)

    list_models = (
        ("evidences", Evidence),
        ("task_results", ResearchDirectionResult),
        ("citations", Citation),
        ("paragraph_bindings", ParagraphBinding),
        ("node_events", NodeEvent),
    )
    for key, model_cls in list_models:
        values = state.get(key)
        if isinstance(values, list):
            state[key] = [
                item if isinstance(item, model_cls) else model_cls.model_validate(item)
                for item in values
            ]


def section(state: object, name: str, model_cls: type[_ModelT]) -> _ModelT:
    """读取并恢复一个嵌套状态区，供直接调用节点和 Graph 共同使用。"""
    mapping = cast(MutableMapping[str, object], state)
    value = mapping.get(name)
    if value is None:
        model = model_cls()
        mapping[name] = model
        return model
    if isinstance(value, model_cls):
        return value
    model = model_cls.model_validate(value)
    mapping[name] = model
    return model


def validate_state_invariants(
    current: dict[str, object],
    update: dict[str, object],
) -> None:
    """校验合并后的顶层生命周期状态。

    节点通常只返回增量，因此必须在 ``current + update`` 的视图上校验，
    不能只检查节点自己的返回字典。
    """
    effective = {**current, **update}
    run = effective.get("run")
    if run is None:
        return
    run = _coerce(RunLifecycle, run)
    if run.phase == "failed" and run.error is None:
        raise StateInvariantError("run.phase=failed 时必须提供 run.error。")
    # ``rendering`` is a terminal preparation phase: research may have ended
    # by budget exhaustion and the renderer still needs the reason to explain
    # the incomplete result. It is therefore valid before ``completed``.
    if run.phase not in {"rendering", "completed", "failed"} and run.terminal_reason:
        raise StateInvariantError(f"run.phase={run.phase} 时不能设置 terminal_reason。")
