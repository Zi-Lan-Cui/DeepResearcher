"""图节点命名与路由决策的单一来源"""

from collections.abc import Mapping
from enum import StrEnum


class NodeName(StrEnum):
    ROUTER = "router"
    CLARIFY = "clarify"
    QUICK_ANSWER = "quick_answer"
    SUPERVISOR = "supervisor"
    WRITER = "writer"
    REVIEWER = "review"
    RENDER_FINAL_REPORT = "render_final_report"


# failed：node_runner 接管；rendering：节点自行宣告的提前终止（预算耗尽、
# 即时回答、证据不足、写作失败）；completed：防御性结束。
TERMINAL_PHASES = frozenset({"failed", "rendering", "completed"})

# 允许产出预览的节点；其他节点产出不对外可见，SSE 只发终态。
PREVIEW_CHANNELS = frozenset({NodeName.SUPERVISOR})


def _field(state: Mapping[str, object], name: str, field: str, default: str) -> str:
    """非变更读取路由字段；条件边不能像 section() 那样恢复模型并写回 State。"""
    value = state.get(name)
    if value is None:
        return default
    if isinstance(value, Mapping):
        return str(value.get(field, default))
    return str(getattr(value, field, default))


def route_after(state: Mapping[str, object], normal: str) -> str:
    """唯一路由主干。"""
    if _field(state, "run", "phase", "routing") in TERMINAL_PHASES:
        return NodeName.RENDER_FINAL_REPORT
    return normal


def route_after_router(state: Mapping[str, object]) -> str:
    # "quick_answer" 取的是 RouteDecision 的业务值，不是节点名；两个字段词表不混用。
    return route_after(
        state,
        NodeName.QUICK_ANSWER if state.get("route") == "quick_answer" else NodeName.CLARIFY,
    )


def route_after_clarify(state: Mapping[str, object]) -> str:
    # 澄清完成（含轮次用尽按原问题结束）后一律进 Supervisor；
    # terminal phase 的直达渲染的出口由 route_after 统一处理。
    return route_after(state, NodeName.SUPERVISOR)


def route_after_quick_answer(state: Mapping[str, object]) -> str:
    # quick_answer 也要经过 Writer 渲染免责声明；Writer 的 quick 分支自行声明 rendering。
    return route_after(state, NodeName.WRITER)


def route_after_supervisor(state: Mapping[str, object]) -> str:
    return route_after(state, str(state.get("supervisor_next", NodeName.WRITER)))


def route_after_writer(state: Mapping[str, object]) -> str:
    # Writer 的所有非成功路径（快速回答/证据不足/耗尽）都会把 phase 置为
    # rendering，由主干结束；这里只需要声明“成功草稿去审阅”。
    return route_after(state, NodeName.REVIEWER)


def route_after_reviewer(state: Mapping[str, object]) -> str:
    # 通过、以及没跑出结论的失败，都直接去交付：后者回流只会再烧一轮审阅。
    return route_after(
        state,
        NodeName.RENDER_FINAL_REPORT
        if _field(state, "review", "status", "pending") in {"approved", "failed"}
        else NodeName.SUPERVISOR,
    )
