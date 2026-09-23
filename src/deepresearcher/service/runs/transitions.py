"""Run 生命周期迁移的唯一事实来源（service 外壳层）。

行按"命名迁移"组织:键是动因的名字(claim、settle_cancelled…),值是
合法来源集合 -> 目标状态 -> 固定原因。同为落到 running 的动因,合法来源并不
相同(claim 不得触碰 awaiting_input 行,生命周期自愈可以),所以"来源->目标"
的二元表装不下这种区别,必须逐个动因命名。不叫 Command:langchain 的
Command 已在仓库里占走这个词,且 reap/settle 是清扫判定、并非谁在下令——
transition 更贴近事实,也与 IllegalTransitionError 同一词根。

两种执行机制共用这份数据:

- 条件 UPDATE(claim/resume 等竞争路径):从表取来源喂 WHERE,写与判定仍在
  同一条语句内原子完成,本模块不接管执行;
- 锁内 ORM 写(settle/reap/cancel 等):apply_transition 在已加载行上按该迁移
  自己的来源集校验再赋值——校验严于"目标并集",用错迁移名同样炸。

side_effects 只收"目标状态唯一决定的值"(lease/finished_at/resume_payload);
terminal_reason 只在它由动因纯决定时随迁移行固定——reap 与 release 是同一条
边、仅 reason 不同,恰为反例证明 reason 不是 f(状态)。引擎产出的动态原因
(finish/fail)与用户文案(error_message)不经表,由调用点覆写。两类东西
刻意不进表:所有权围栏(lease_owner/attempt 谓词)回答"谁能推",不是"能
不能推";无 claim 的对账写(执行器把状态机真相回写行里)只受 may_overwrite
"不得覆盖终态"约束——那不是命名迁移。
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


@dataclass(frozen=True)
class Transition:
    """一条命名迁移:合法来源 -> 目标状态;reason 是该动因固定携带的终态原因。"""

    target: str
    sources: frozenset[str]
    reason: str | None = None


# 动因名 -> 迁移
TRANSITIONS: dict[str, Transition] = {
    # 领取:FIFO 只动 queued;resume 领取额外允许被 reap 的 interrupted。
    "claim": Transition("running", frozenset({"queued"})),
    "claim_resume": Transition("running", frozenset({"queued", "interrupted"})),
    # 生命周期自愈:执行器无 claim 地把行拉回 running(awaiting_input 仅在
    # 与用户回答竞态的残余窗口里出现,claim 路径永远看不到它)。
    "mark_running": Transition("running", frozenset({"queued", "interrupted", "awaiting_input"})),
    # 引擎暂停等待澄清:释放租约但保留行,等待 answer_resume。
    "await_input": Transition("awaiting_input", frozenset({"running"})),
    # 心跳/领取者让位:reap 由清扫者代做(判死),release 由持有者自做(shutdown)。
    "reap": Transition("interrupted", frozenset({"running"}), reason="lease_expired"),
    "release": Transition("interrupted", frozenset({"running"}), reason="server_shutdown"),
    # 用户提交澄清回答:awaiting_input 重新入队。
    "answer_resume": Transition("queued", frozenset({"awaiting_input"})),
    # 取消的三条路:未领取行立即结算;running 由执行器收到信号后自写;
    # "接了意图却没写终态就死了"的 interrupted 由清扫者代笔。
    "cancel_pending": Transition(
        "cancelled", frozenset({"queued", "awaiting_input", "interrupted"}), reason="user_cancelled"
    ),
    "cancel_running": Transition("cancelled", frozenset({"running"}), reason="user_cancelled"),
    "settle_cancelled": Transition(
        "cancelled", frozenset({"interrupted"}), reason="user_cancelled"
    ),
    # 正常终态只能由 running 的执行者写出;reason 来自引擎产物,不经表。
    "finish": Transition("completed", frozenset({"running"})),
    "fail": Transition("failed", frozenset({"running"})),
    # 启动恢复:无租约的 running/上轮遗留的 interrupted 且无 checkpoint → 判死。
    "recover_dead": Transition(
        "failed", frozenset({"running", "interrupted"}), reason="server_restart"
    ),
}


class IllegalTransitionError(RuntimeError):
    """任何命名迁移的来源集之外的改状态——状态机的 bug,不是运行时数据问题。"""


def transition_for(name: str) -> Transition:
    return TRANSITIONS[name]


_LEGAL: dict[str, frozenset[str]] = {}
for _transition in TRANSITIONS.values():
    _LEGAL[_transition.target] = _LEGAL.get(_transition.target, frozenset()) | _transition.sources


def is_legal(source: str, target: str) -> bool:
    """这条箭头原则上存在吗(不分动因,取该目标所有迁移来源的并集)。"""
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

    校验用该迁移自己的来源集:箭头存在但用错动因名(如拿 cancel_pending 推
    running 行)同样拒绝。迁移固定携带的 reason 在结构副作用之后写入。
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
    """无 claim 的对账写唯一纪律:不覆盖终态。"""
    return status not in TERMINAL_STATUSES
