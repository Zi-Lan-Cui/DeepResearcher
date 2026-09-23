"""Run 状态机命名迁移表的独立校验。

EXPECT_* 是手抄自 service/runs/transitions.py 的第二份表:表改动时这里必须同步。
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
    may_overwrite,
    side_effects,
    transition_for,
)

NOW = datetime(2026, 9, 23, 12, 0, 0, tzinfo=timezone.utc)

EXPECTED_TRANSITIONS: dict[str, tuple[str, frozenset[str], str | None]] = {
    "claim": ("running", frozenset({"queued"}), None),
    "claim_resume": ("running", frozenset({"queued", "interrupted"}), None),
    "mark_running": ("running", frozenset({"queued", "interrupted", "awaiting_input"}), None),
    "await_input": ("awaiting_input", frozenset({"running"}), None),
    "reap": ("interrupted", frozenset({"running"}), "lease_expired"),
    "release": ("interrupted", frozenset({"running"}), "server_shutdown"),
    "answer_resume": ("queued", frozenset({"awaiting_input"}), None),
    "cancel_pending": (
        "cancelled",
        frozenset({"queued", "awaiting_input", "interrupted"}),
        "user_cancelled",
    ),
    "cancel_running": ("cancelled", frozenset({"running"}), "user_cancelled"),
    "settle_cancelled": ("cancelled", frozenset({"interrupted"}), "user_cancelled"),
    "finish": ("completed", frozenset({"running"}), None),
    "fail": ("failed", frozenset({"running"}), None),
    "recover_dead": ("failed", frozenset({"running", "interrupted"}), "server_restart"),
}

# 来源并集:每个目标状态的全部合法来源。
EXPECTED_LEGAL: dict[str, frozenset[str]] = {
    "running": frozenset({"queued", "interrupted", "awaiting_input"}),
    "awaiting_input": frozenset({"running"}),
    "interrupted": frozenset({"running"}),
    "queued": frozenset({"awaiting_input"}),
    "cancelled": frozenset({"queued", "running", "awaiting_input", "interrupted"}),
    "completed": frozenset({"running"}),
    "failed": frozenset({"running", "interrupted"}),
}

TERMINALS = frozenset({"completed", "failed", "cancelled"})


def test_transition_table_matches_hand_copied_expectations() -> None:
    actual = {
        name: (transition.target, transition.sources, transition.reason)
        for name, transition in TRANSITIONS.items()
    }
    assert actual == EXPECTED_TRANSITIONS


def test_terminal_states_are_sinks() -> None:
    for terminal in TERMINALS:
        assert all(not is_legal(terminal, target) for target in ALL_STATUSES)
        for transition in TRANSITIONS.values():
            assert terminal not in transition.sources


def test_reason_is_carried_by_fixed_provenance_transitions() -> None:
    # reap/release 是同一条边、仅 reason 不同的对照组:证明 reason 不是 f(状态)。
    assert TRANSITIONS["reap"].target == TRANSITIONS["release"].target
    assert TRANSITIONS["reap"].sources == TRANSITIONS["release"].sources
    # 两边都必须带固定原因:只查"互异"会放过 None != "server_shutdown" 的情况。
    assert TRANSITIONS["reap"].reason is not None
    assert TRANSITIONS["release"].reason is not None
    assert TRANSITIONS["reap"].reason != TRANSITIONS["release"].reason
    # 引擎动态原因(终态产物)不随行固定。
    assert TRANSITIONS["finish"].reason is None
    assert TRANSITIONS["fail"].reason is None


def test_no_self_transitions() -> None:
    for status in ALL_STATUSES:
        assert not is_legal(status, status)


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
    assert terminal["lease_expires_at"] is None
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
    assert "resume_payload" not in interrupted  # 中断行保留 payload,供 resume 领取
    assert "terminal_reason" not in interrupted  # 随迁移行固定(reap/release),不经状态

    assert side_effects("running", now=NOW) == {"status": "running"}
    assert side_effects("queued", now=NOW) == {"status": "queued"}


def _row(status: str) -> SimpleNamespace:
    return SimpleNamespace(
        status=status,
        finished_at=None,
        lease_owner="worker-1",
        lease_expires_at=NOW,
        resume_payload={"answer": "a"},
        terminal_reason="old",
        error_message="old",
    )


def test_apply_transition_fills_fixed_reason() -> None:
    row = _row("running")
    apply_transition(row, "reap", now=NOW)
    assert row.status == "interrupted"
    assert row.terminal_reason == "lease_expired"
    assert row.lease_owner is None
    assert row.resume_payload == {"answer": "a"}  # interrupted 保留在途回答

    settled = _row("interrupted")
    apply_transition(settled, "settle_cancelled", now=NOW)
    assert settled.terminal_reason == "user_cancelled"
    assert settled.finished_at == NOW
    assert settled.resume_payload is None


def test_apply_transition_rejects_wrong_name_even_when_edge_exists() -> None:
    # running→cancelled 这条边存在(cancel_running),但 cancel_pending 不许推它:
    # 按迁移名校验严于按并集校验。
    row = _row("running")
    doomed_before = vars(row).copy()
    with pytest.raises(IllegalTransitionError):
        apply_transition(row, "cancel_pending", now=NOW)
    assert vars(row) == doomed_before  # 校验先于赋值,失败不留半状态


def test_apply_transition_rejects_illegal_source() -> None:
    doomed = _row("cancelled")
    with pytest.raises(IllegalTransitionError):
        apply_transition(doomed, "mark_running", now=NOW)
    assert doomed.status == "cancelled"


def test_transition_for_exposes_table_rows() -> None:
    assert transition_for("claim") == Transition("running", frozenset({"queued"}))


def test_may_overwrite_only_shields_terminals() -> None:
    assert all(may_overwrite(status) for status in ALL_STATUSES if status not in TERMINALS)
    assert not any(may_overwrite(status) for status in TERMINALS)
