"""Run 状态机迁移表的独立校验。

EXPECT_* 是从 service/runs/transitions.py 手抄的第二份表;迁移表改动时这里必须同步。
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from deepresearcher.service.runs.transitions import (
    ALL_STATUSES,
    TRANSITIONS,
    IllegalTransitionError,
    Transition,
    apply_transition,
    assert_transition,
    is_legal,
    side_effects,
    transition_for,
)

NOW = datetime(2026, 9, 23, 12, 0, 0, tzinfo=timezone.utc)

EXPECTED_TRANSITIONS: dict[str, tuple[str, frozenset[str], str | None]] = {
    "claim_resume": ("running", frozenset({"queued", "interrupted"}), None),
    "adopt": ("running", frozenset({"running"}), None),
    "await_input": ("awaiting_input", frozenset({"running"}), None),
    "release": ("interrupted", frozenset({"running"}), "server_shutdown"),
    "answer_resume": ("queued", frozenset({"awaiting_input"}), None),
    "cancel_pending": (
        "cancelled",
        frozenset({"queued", "awaiting_input", "interrupted"}),
        "user_cancelled",
    ),
    "cancel_running": ("cancelled", frozenset({"running"}), "user_cancelled"),
    "finish": ("completed", frozenset({"running"}), None),
    "fail": ("failed", frozenset({"running"}), None),
    "settle_exhausted": (
        "failed",
        frozenset({"queued", "interrupted", "running"}),
        "dispatch_attempts_exhausted",
    ),
    "reopen_exhausted": ("interrupted", frozenset({"failed"}), None),
}

# 来源并集：每个目标状态的全部合法来源。
EXPECTED_LEGAL: dict[str, frozenset[str]] = {
    "running": frozenset({"queued", "interrupted", "running"}),
    "awaiting_input": frozenset({"running"}),
    "interrupted": frozenset({"running", "failed"}),
    "queued": frozenset({"awaiting_input"}),
    "cancelled": frozenset({"queued", "running", "awaiting_input", "interrupted"}),
    "completed": frozenset({"running"}),
    "failed": frozenset({"running", "interrupted", "queued"}),
}

TERMINALS = frozenset({"completed", "failed", "cancelled"})


def test_transition_table_matches_hand_copied_expectations() -> None:
    actual = {
        name: (transition.target, transition.sources, transition.reason)
        for name, transition in TRANSITIONS.items()
    }
    assert actual == EXPECTED_TRANSITIONS


def test_terminal_states_are_sinks_except_explicit_recovery() -> None:
    """终态不自己往外走，唯一例外是运维显式恢复。

    「终态是吸收态」这条保护的实质是：任何自动路径都不能把已经结案的 run 拉回执行。
    reopen_exhausted 是唯一例外：它只能由运维入口调用，且只能拉起
    dispatch_attempts_exhausted 结案的行。新增的自动迁移若从这个白名单外
    指回终态，本用例仍然会失败。
    """
    allowed = {"reopen_exhausted"}
    for terminal in TERMINALS:
        for name, transition in TRANSITIONS.items():
            if name in allowed:
                continue
            assert terminal not in transition.sources, f"{name} 从终态 {terminal} 出发"
        for target in ALL_STATUSES:
            if is_legal(terminal, target):
                assert any(terminal in TRANSITIONS[name].sources for name in allowed), (
                    f"终态 {terminal} 出边只允许来自显式恢复"
                )


def test_reason_is_carried_by_fixed_provenance_transitions() -> None:
    # cancel_pending 与 cancel_running 目标相同、来源不同、reason 相同：
    # 说明 reason 由动因决定，不是状态本身的函数。
    assert TRANSITIONS["cancel_pending"].target == TRANSITIONS["cancel_running"].target
    assert TRANSITIONS["cancel_pending"].sources != TRANSITIONS["cancel_running"].sources
    assert TRANSITIONS["cancel_pending"].reason == TRANSITIONS["cancel_running"].reason
    assert TRANSITIONS["cancel_pending"].reason is not None
    # 死信结案必须自带原因：它和普通 fail 目标相同，只靠 reason 区分是谁结的案。
    assert TRANSITIONS["settle_exhausted"].target == TRANSITIONS["fail"].target
    assert TRANSITIONS["settle_exhausted"].reason != TRANSITIONS["fail"].reason
    # finish/fail 的 reason 由引擎按运行结果填写，迁移表中不固定。
    assert TRANSITIONS["finish"].reason is None
    assert TRANSITIONS["fail"].reason is None


def test_only_adopt_is_a_deliberate_self_transition() -> None:
    """唯一的自转移是 adopt，且它的合法性由「消息到达」证明。

    自转移不能用来源集表达「上一个持有者已停止续期」，那件事的证据是消息重投
    这个事件本身；因此状态机允许 running→running，但只有这一条动因可以。
    其余任何状态自转都说明表写错了。
    """
    self_transitions = [
        name
        for name, transition in TRANSITIONS.items()
        if transition.sources == {transition.target}
    ]
    assert self_transitions == ["adopt"]
    assert is_legal("running", "running")


@pytest.mark.parametrize("source", ALL_STATUSES)
@pytest.mark.parametrize("target", ALL_STATUSES)
def test_full_matrix(source: str, target: str) -> None:
    expected = source in EXPECTED_LEGAL.get(target, frozenset())
    assert is_legal(source, target) is expected
    if expected:
        assert_transition(source, target)
    else:
        with pytest.raises(IllegalTransitionError):
            assert_transition(source, target)


def test_side_effects_terminal_and_intermediate() -> None:
    terminal = side_effects("cancelled", now=NOW)
    assert terminal["status"] == "cancelled"
    assert terminal["finished_at"] == NOW
    assert terminal["lease_owner"] is None
    assert terminal["resume_payload"] is None

    awaiting = side_effects("awaiting_input", now=NOW)
    assert awaiting["finished_at"] is None
    assert awaiting["lease_owner"] is None
    assert awaiting["resume_payload"] is None  # 回答后走 answer_resume 重新携带
    assert awaiting["terminal_reason"] is None
    assert awaiting["error_message"] is None

    interrupted = side_effects("interrupted", now=NOW)
    assert interrupted["lease_owner"] is None
    assert interrupted["finished_at"] is None
    assert "resume_payload" not in interrupted  # 中断行保留 payload，供 resume 领取
    assert "terminal_reason" not in interrupted  # 由迁移行自带的 reason 写入，不在副作用里填

    assert side_effects("running", now=NOW) == {"status": "running"}
    assert side_effects("queued", now=NOW) == {"status": "queued"}


def _row(status: str) -> SimpleNamespace:
    return SimpleNamespace(
        status=status,
        finished_at=None,
        lease_owner="worker-1",
        resume_payload={"answer": "a"},
        terminal_reason="old",
        error_message="old",
    )


def test_apply_transition_fills_fixed_reason() -> None:
    row = _row("running")
    apply_transition(row, "release", now=NOW)
    assert row.status == "interrupted"
    assert row.terminal_reason == "server_shutdown"
    assert row.lease_owner is None
    assert row.resume_payload == {"answer": "a"}  # interrupted 保留在途回答

    settled = _row("interrupted")
    apply_transition(settled, "settle_exhausted", now=NOW)
    assert settled.terminal_reason == "dispatch_attempts_exhausted"
    assert settled.finished_at == NOW
    assert settled.resume_payload is None


def test_apply_transition_rejects_wrong_name_even_when_edge_exists() -> None:
    # running→cancelled 这条边经 cancel_running 存在，但 cancel_pending 不能用它：
    # 按迁移名校验严于按来源并集校验。
    row = _row("running")
    doomed_before = vars(row).copy()
    with pytest.raises(IllegalTransitionError):
        apply_transition(row, "cancel_pending", now=NOW)
    assert vars(row) == doomed_before  # 校验先于赋值，失败不留半状态


def test_apply_transition_rejects_illegal_source() -> None:
    # cancelled 是终态，任何自动迁移都不能从这里出发。
    doomed = _row("cancelled")
    with pytest.raises(IllegalTransitionError):
        apply_transition(doomed, "claim_resume", now=NOW)
    assert doomed.status == "cancelled"


def test_transition_for_exposes_table_rows() -> None:
    assert transition_for("claim_resume") == Transition(
        "running", frozenset({"queued", "interrupted"})
    )
