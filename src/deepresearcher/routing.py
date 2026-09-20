"""图节点命名与路由决策的单一来源。

节点名是跨模块契约：graph.py 用它挂边，node_runner 与 Supervisor 把
``supervisor_next`` 写进 State。因此常量必须放在 ``nodes/`` 与
``graph.py`` 之外的中性模块，否则叶子 Agent 又要反向 import 装配层。

路由规则只有一条主干（``route_after``）：节点一旦声明 terminal phase，
无条件收束到渲染终点；否则跟随该节点写下的业务交接。各节点的出口
函数是主干 + 各自一个字段的业务判断，路由表整体在本文件可一眼读尽。
"""

from collections.abc import Mapping
from enum import StrEnum


class NodeName(StrEnum):
    ROUTER = "router"
    CLARIFY = "clarify"
    QUICK_ANSWER = "quick_answer"
    SUPERVISOR = "supervisor"
    WRITER = "writer"
    REVIEWER = "reflection"
    RENDER_FINAL_REPORT = "render_final_report"


# failed：node_runner 接管；rendering：节点自行宣告的提前终止（预算耗尽、
# 即时回答、证据不足、写作失败）；completed：防御性收束。
TERMINAL_PHASES = frozenset({"failed", "rendering", "completed"})


def _field(state: Mapping[str, object], name: str, field: str, default: str) -> str:
    """非变更读取路由字段；条件边不能像 ``section()`` 那样恢复模型并写回 State。"""
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
    # "quick_answer" 是 RouteDecision 的业务取值，不是节点名，两个词汇空间不混用。
    return route_after(
        state,
        NodeName.QUICK_ANSWER if state.get("route") == "quick_answer" else NodeName.CLARIFY,
    )


def route_after_clarify(state: Mapping[str, object]) -> str:
    return route_after(
        state,
        NodeName.RENDER_FINAL_REPORT
        if state.get("answer_mode") == "clarification_needed"
        else NodeName.SUPERVISOR,
    )


def route_after_quick_answer(state: Mapping[str, object]) -> str:
    # quick_answer 也要经过 Writer 渲染免责声明；Writer 的 quick 分支自行声明 rendering。
    return route_after(state, NodeName.WRITER)


def route_after_supervisor(state: Mapping[str, object]) -> str:
    return route_after(state, str(state.get("supervisor_next", NodeName.WRITER)))


def route_after_writer(state: Mapping[str, object]) -> str:
    # Writer 的所有非成功路径（快速回答/证据不足/耗尽）都会把 phase 置为
    # rendering，由主干收束；这里只需要声明“成功草稿去审阅”。
    return route_after(state, NodeName.REVIEWER)


def route_after_reviewer(state: Mapping[str, object]) -> str:
    return route_after(
        state,
        NodeName.RENDER_FINAL_REPORT
        if _field(state, "review", "status", "pending") == "approved"
        else NodeName.SUPERVISOR,
    )
