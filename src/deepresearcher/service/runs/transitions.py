"""Run 生命周期迁移表，规定状态间合法的迁移关系。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from deepresearcher.service.storage.tables import TERMINAL_STATUSES

ALL_STATUSES = (
    "queued",
    "running",
    "awaiting_input",
    "interrupted",
    *TERMINAL_STATUSES,
)

OUTSTANDING_STATUSES = tuple(sorted(frozenset(ALL_STATUSES) - frozenset(TERMINAL_STATUSES)))


@dataclass(frozen=True)
class Transition:
    """一条命名迁移：合法来源 -> 目标状态；reason 是该动因固定携带的终态原因。"""

    target: str
    sources: frozenset[str]
    reason: str | None = None


# 动因名 -> 迁移
TRANSITIONS: dict[str, Transition] = {
    # 领取：queued 首跑；interrupted 是优雅关停后重投，两种都按续跑处理。
    "claim_resume": Transition("running", frozenset({"queued", "interrupted"})),
    # 接管：消息重投后行仍是 running，说明上一持有者已停止续期
    "adopt": Transition("running", frozenset({"running"})),
    # 引擎暂停等待澄清：释放所有权但保留行，等待 answer_resume。
    "await_input": Transition("awaiting_input", frozenset({"running"})),
    # 落到 interrupted 的唯一来源：持有者优雅关停时主动让位。暴力崩溃不留状态，
    # 由可见性到期后的重投接管。
    "release": Transition("interrupted", frozenset({"running"}), reason="server_shutdown"),
    # 用户提交澄清回答：awaiting_input 重新入队。
    "answer_resume": Transition("queued", frozenset({"awaiting_input"})),
    # 取消的两条路：没有执行者时直接结算；running 由收到取消信号的一方结算，
    # 或由执行器自己写终态。
    "cancel_pending": Transition(
        "cancelled", frozenset({"queued", "awaiting_input", "interrupted"}), reason="user_cancelled"
    ),
    "cancel_running": Transition("cancelled", frozenset({"running"}), reason="user_cancelled"),
    # 正常终态只能由 running 的执行者写出；reason 来自引擎产物，不经表。
    "finish": Transition("completed", frozenset({"running"})),
    "fail": Transition("failed", frozenset({"running"})),
    # 入站死信：同一条分派消息投递次数耗尽仍未跑起来。running 也在范围内——投递
    # 次数涨到上限意味着持有者连续多轮都没续上可见性，它已经不在了。
    "settle_exhausted": Transition(
        "failed",
        frozenset({"queued", "interrupted", "running"}),
        reason="dispatch_attempts_exhausted",
    ),
    # 死信的显式恢复：failed(dispatch_attempts_exhausted) → interrupted 等重投。
    "reopen_exhausted": Transition("interrupted", frozenset({"failed"})),
}


class IllegalTransitionError(RuntimeError):
    """任何命名迁移的来源集之外的改状态——状态机的 bug，不是运行时数据问题。"""


def transition_for(name: str) -> Transition:
    return TRANSITIONS[name]


_LEGAL: dict[str, frozenset[str]] = {}
for _transition in TRANSITIONS.values():
    _LEGAL[_transition.target] = _LEGAL.get(_transition.target, frozenset()) | _transition.sources


def is_legal(source: str, target: str) -> bool:
    """这条箭头原则上存在吗(不分动因，取该目标所有迁移来源的并集)。"""
    return source in _LEGAL.get(target, frozenset())


def assert_transition(source: str, target: str) -> None:
    if not is_legal(source, target):
        raise IllegalTransitionError(f"illegal run status transition: {source} -> {target}")


def side_effects(target: str, *, now: datetime) -> dict[str, Any]:
    """落到目标状态时必须一致的结构字段。"""
    effects: dict[str, Any] = {"status": target}
    if target in TERMINAL_STATUSES:
        effects.update(finished_at=now, lease_owner=None, resume_payload=None)
    elif target == "awaiting_input":
        effects.update(
            terminal_reason=None,
            error_message=None,
            finished_at=None,
            lease_owner=None,
            resume_payload=None,
        )
    elif target == "interrupted":
        effects.update(lease_owner=None, finished_at=None)
    return effects


def apply_transition(row: Any, name: str, *, now: datetime) -> None:
    """在已加载(且调用方保证已持锁或处于单写路径)的行上执行一条命名迁移。"""
    transition = TRANSITIONS[name]
    if row.status not in transition.sources:
        raise IllegalTransitionError(
            f"transition {name!r} cannot fire from status {row.status!r}"
            f" (allowed: {sorted(transition.sources)})"
        )
    for key, value in side_effects(transition.target, now=now).items():
        setattr(row, key, value)
    if transition.reason is not None:
        row.terminal_reason = transition.reason
