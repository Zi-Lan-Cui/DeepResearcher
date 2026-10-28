"""LangGraph 顶层状态"""

from collections.abc import MutableMapping
from operator import add
from typing import (
    Annotated,
    TypedDict,
    TypeVar,
    cast,
    get_args,
    get_type_hints,
)

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from deepresearcher.errors import AgentError
from deepresearcher.evidence.models import Evidence
from deepresearcher.routing import NodeName
from deepresearcher.schemas import (
    Citation,
    ParagraphBinding,
    ResearchSynthesis,
    ReviewProgress,
    RunStatus,
    SupervisorProgress,
    TaskResult,
    WriterDirective,
    WriterProgress,
)
from deepresearcher.vocab import AnswerMode


class StateInvariantError(AgentError):
    """节点产生了互相矛盾的运行状态。"""

    code = "invalid_state"


_ModelT = TypeVar("_ModelT", bound=BaseModel)


def _coerce(model_cls: type[_ModelT], value: object) -> _ModelT:
    """跨进程 checkpoint 恢复时值可能是 dict；恢复为模型，否则原样返回。"""
    if isinstance(value, model_cls):
        return value
    return model_cls.model_validate(value)


def merge_evidences(current: list[Evidence], incoming: list[Evidence]) -> list[Evidence]:
    """按 evidence_id 去重合并；同 id 而内容不同视为状态契约冲突，抛出 StateInvariantError。"""
    restored = [_coerce(Evidence, item) for item in [*current, *incoming]]
    by_id: dict[str, Evidence] = {}
    for item in restored:
        previous = by_id.get(item.evidence_id)
        if previous is not None and previous != item:
            raise StateInvariantError(f"Evidence ID 冲突且内容不同：{item.evidence_id}。")
        by_id[item.evidence_id] = item
    return list(by_id.values())


def merge_task_results(
    current: list[TaskResult],
    incoming: list[TaskResult],
) -> list[TaskResult]:
    """按 task_id 去重合并；同一 task 因恢复而重试时，以新结果覆盖旧结果。"""
    restored = [_coerce(TaskResult, item) for item in [*current, *incoming]]
    by_id: dict[str, TaskResult] = {}
    for item in restored:
        by_id[item.task_id] = item
    return list(by_id.values())


def merge_unique(current: list[str], incoming: list[str]) -> list[str]:
    """字符串集合语义合并，保持首次出现顺序。"""
    return list(dict.fromkeys([*current, *incoming]))


class RunState(TypedDict, total=False):
    """一次 run 的跨节点状态容器：图编译时的 state_schema，thread_id 即 run_id。"""

    # State 只保存数据快照；Agent、LLM、工具和锁由 graph.py 装配时创建并持有。
    run: RunStatus
    supervisor: SupervisorProgress
    writer: WriterProgress
    review: ReviewProgress
    run_id: str

    # 输入与路由字段
    query: str
    clarified_query: str
    session_id: str
    route: str
    route_reason: str
    answer_mode: AnswerMode
    # 记录用户意图的摘要，供后续节点参考；不作为报告正文内容使用。
    intent_summary: str
    assumptions: list[str]
    clarification_log: Annotated[list[dict[str, object]], add]
    draft_answer: str

    supervisor_messages: Annotated[list[BaseMessage], add]
    evidences: Annotated[list[Evidence], merge_evidences]
    # 已读取来源的 URL 快照引用，不作为报告正文内容使用。
    source_refs: Annotated[list[str], merge_unique]
    # Supervisor 当前工作集；完整 Evidence 档案仍保存在 evidences 中。
    active_evidence_ids: list[str]
    working_set_revision: int
    research_synthesis: ResearchSynthesis | None
    writer_directive: WriterDirective | None
    task_results: Annotated[list[TaskResult], merge_task_results]
    supervisor_next: NodeName

    # Writer 未通过校验时的备用草稿，exhausted 或 citation_protocol 失败时写入；
    # 区别于 report_draft，rejection 回流时作为 Supervisor 的修订底稿。
    rejected_draft: str
    # Writer 产出的草稿，以 [[cite:evidence_id]] 标记引用；
    # 审阅通过后由终检渲染层编号并渲染为 report。
    report_draft: str
    # run_headline 是对问题的一句话压缩
    run_headline: str
    # report_title 是 Writer 为回答拟的文章标题
    report_title: str
    paragraph_bindings: list[ParagraphBinding]
    citations: list[Citation]
    report: str

    # 审计计数字段
    evidence_count: int
    source_count: int


_OPTIONAL_SCALAR_KEYS = frozenset(
    key for key, hint in get_type_hints(RunState).items() if type(None) in get_args(hint)
)


def restore_state_models(state: dict[str, object]) -> None:
    """恢复 JSON checkpoint 中被还原为 dict 的嵌套模型。"""
    scalar_models = (
        ("run", RunStatus),
        ("supervisor", SupervisorProgress),
        ("writer", WriterProgress),
        ("review", ReviewProgress),
        ("writer_directive", WriterDirective),
        ("research_synthesis", ResearchSynthesis),
    )
    for key, model_cls in scalar_models:
        value = state.get(key)
        if value is None:
            if key in _OPTIONAL_SCALAR_KEYS:
                continue
            state[key] = model_cls()  # type: ignore[call-arg]
        elif not isinstance(value, model_cls):
            state[key] = model_cls.model_validate(value)

    list_models = (
        ("evidences", Evidence),
        ("task_results", TaskResult),
        ("citations", Citation),
        ("paragraph_bindings", ParagraphBinding),
    )
    for key, model_cls in list_models:
        values = state.get(key)
        if isinstance(values, list):
            state[key] = [
                item if isinstance(item, model_cls) else model_cls.model_validate(item)
                for item in values
            ]


def section(state: object, name: str, model_cls: type[_ModelT]) -> _ModelT:
    """读取并恢复一个嵌套状态区；缺失时写入默认实例，dict 形态恢复为模型后写回，
    调用方与 state 共享同一对象。
    """
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

    节点通常只返回增量，因此必须在 current + update 的视图上校验，
    不能只检查节点自己的返回字典。
    """
    effective = {**current, **update}
    run = effective.get("run")
    if run is None:
        return
    run = _coerce(RunStatus, run)
    if run.phase == "failed" and run.error is None:
        raise StateInvariantError("run.phase=failed 时必须提供 run.error。")
    # rendering 是进入终点的准备阶段：因预算耗尽收尾时，渲染层需要 terminal_reason
    # 解释报告为何不完整，所以该阶段携带 terminal_reason 是合法的。
    if run.phase not in {"rendering", "completed", "failed"} and run.terminal_reason:
        raise StateInvariantError(f"run.phase={run.phase} 时不能设置 terminal_reason。")
