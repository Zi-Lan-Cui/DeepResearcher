"""Run 生命周期迁移表：动因名 → 合法来源集 → 目标状态 → 固定原因。

两种写路径共用本表：条件 UPDATE 把来源集写进 WHERE 子句，判定与写入在同一条语句内
完成；锁内 ORM 写由 apply_transition 按该迁移自己的来源集校验后赋值。
side_effects 只放目标状态唯一决定的值；terminal_reason 仅在由动因纯决定时随行固定，
其余由调用点覆写。lease_owner 与 fence_token 是所有权谓词，不进表。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from deepresearcher.service.persistence.models import TERMINAL_STATUSES

ALL_STATUSES = (
    "queued",
    "running",
    "awaiting_input",
    "interrupted",
    *TERMINAL_STATUSES,
)

# per-user 并发配额计数的状态集合：用户已受理且未完成的行。
# awaiting_input 必须计入：answer_resume 把它 CAS 回 queued 时不重过准入检查，
# 不计入则可以先建立一批待澄清行、再逐一回答来绕过上限。
OUTSTANDING_STATUSES = tuple(sorted(frozenset(ALL_STATUSES) - frozenset(TERMINAL_STATUSES)))


@dataclass(frozen=True)
class Transition:
    """一条命名迁移：合法来源 -> 目标状态；reason 是该动因固定携带的终态原因。"""

    target: str
    sources: frozenset[str]
    reason: str | None = None


# 动因名 -> 迁移
TRANSITIONS: dict[str, Transition] = {
    # 领取：FIFO 只动 queued;resume 领取额外允许被 reap 的 interrupted。
    "claim": Transition("running", frozenset({"queued"})),
    "claim_resume": Transition("running", frozenset({"queued", "interrupted"})),
    # 生命周期自动恢复：执行器无 claim 地把行回滚 running。awaiting_input 只在
    # 与用户回答竞态的残余窗口出现，claim 路径不会命中该状态。
    "mark_running": Transition("running", frozenset({"queued", "interrupted", "awaiting_input"})),
    # 引擎暂停等待澄清：释放租约但保留行，等待 answer_resume。
    "await_input": Transition("awaiting_input", frozenset({"running"})),
    # 落到 interrupted 的两条路：reap 由清理扫描补写(lease_expired),
    # release 由持有者主动让位(server_shutdown)。
    "reap": Transition("interrupted", frozenset({"running"}), reason="lease_expired"),
    "release": Transition("interrupted", frozenset({"running"}), reason="server_shutdown"),
    # 用户提交澄清回答：awaiting_input 重新入队。
    "answer_resume": Transition("queued", frozenset({"awaiting_input"})),
    # 取消的三条路：未领取行立即结算；running 由执行器收到信号后自写；
    # executor 退出遗留的 interrupted 行由 settle_cancelled 补写终态。
    "cancel_pending": Transition(
        "cancelled", frozenset({"queued", "awaiting_input", "interrupted"}), reason="user_cancelled"
    ),
    "cancel_running": Transition("cancelled", frozenset({"running"}), reason="user_cancelled"),
    "settle_cancelled": Transition(
        "cancelled", frozenset({"interrupted"}), reason="user_cancelled"
    ),
    # 正常终态只能由 running 的执行者写出；reason 来自引擎产物，不经表。
    "finish": Transition("completed", frozenset({"running"})),
    "fail": Transition("failed", frozenset({"running"})),
    # 启动恢复：无租约的 running/上轮遗留的 interrupted 且无 checkpoint → 判定不可恢复。
    "recover_dead": Transition(
        "failed", frozenset({"running", "interrupted"}), reason="server_restart"
    ),
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
    """落到目标状态时必须一致的结构字段(列名 -> 值)。

    只收"目标状态唯一决定的值";terminal_reason 是否随行由迁移行决定
    (见 apply_transition),error_message 等文案不经这里。
    """
    effects: dict[str, Any] = {"status": target}
    if target in TERMINAL_STATUSES:
        effects.update(
            finished_at=now, lease_owner=None, lease_expires_at=None, resume_payload=None
        )
    elif target == "awaiting_input":
        effects.update(
            terminal_reason=None,
            error_message=None,
            finished_at=None,
            lease_owner=None,
            lease_expires_at=None,
            resume_payload=None,
        )
    elif target == "interrupted":
        effects.update(lease_owner=None, lease_expires_at=None, finished_at=None)
    return effects


def apply_transition(row: Any, name: str, *, now: datetime) -> None:
    """在已加载(且调用方保证已持锁或处于单写路径)的行上执行一条命名迁移。

    校验用该迁移自己的来源集：箭头存在但用错动因名(如拿 cancel_pending 推
    running 行)同样拒绝。迁移固定携带的 reason 在结构副作用之后写入。

    参数：
        row: 已加载的 Run 行(鸭子类型)。
        name: 动因名。
        now: 用于终态字段的时间戳。

    抛出：
        IllegalTransitionError: 当前状态不在该迁移来源集内时。
    """
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


def may_overwrite(status: str) -> bool:
    """无 claim 的对账写唯一纪律：不覆盖终态。"""
    return status not in TERMINAL_STATUSES
