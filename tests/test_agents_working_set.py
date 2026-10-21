"""共享 working-set 原语的契约测试。"""

from deepresearcher.agents.working_set import release_working_set, restore_working_set
from deepresearcher.evidence.models import Evidence, normalize_evidence_ids
from deepresearcher.schemas.reporting import ResearchAspect
from deepresearcher.schemas.tool_args import (
    CompleteReport,
    ReadEvidence,
    ReleaseEvidence,
    ResearchDirectionComplete,
    RestoreEvidence,
)


def _evidence(evidence_id: str) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        subtask_id="t1",
        research_direction="方向",
        claim="断言",
        quote="原文",
        source_url="https://example.com/a",
        source_title="A",
    )


class FakeWorkingSet:
    """最小 LoopState 替身:只含档案与活跃集,提供 release/restore 两个转移方法。"""

    def __init__(self, archive_ids: list[str], active_ids: list[str]) -> None:
        self.evidences = [_evidence(item) for item in archive_ids]
        self.active_evidence_ids = set(active_ids)

    def release_evidence(self, evidence_ids: list[str]) -> list[str]:
        existing = self.active_evidence_ids.intersection(evidence_ids)
        self.active_evidence_ids.difference_update(existing)
        return sorted(existing)

    def restore_evidence(self, evidence_ids: list[str]) -> list[str]:
        archived = {item.evidence_id for item in self.evidences}
        restored = [item for item in dict.fromkeys(evidence_ids) if item in archived]
        self.active_evidence_ids.update(restored)
        return restored


def test_release_receipt_has_three_keys_and_post_mutation_snapshot():
    state = FakeWorkingSet(["e1", "e2"], ["e1", "e2"])
    receipt = release_working_set(
        state, ["e1"], snapshot=lambda: {"active_evidence_count": len(state.active_evidence_ids)}
    )
    assert receipt["released_evidence_ids"] == ["e1"]
    assert receipt["not_in_working_set_ids"] == []
    assert receipt["unknown_evidence_ids"] == []
    # 快照必须在释放之后求值：活跃数应已减 1。
    assert receipt["active_evidence_count"] == 1


def test_duplicate_and_unknown_ids_classified_consistently_on_both_ops():
    release_state = FakeWorkingSet(["e1"], ["e1"])
    receipt = release_working_set(release_state, ["e1", "e1", "ghost"], snapshot=lambda: {})
    # 请求按序去重，重复 id 不在任何键里出现两次；ghost 不在档案中，归入 unknown。
    assert receipt["released_evidence_ids"] == ["e1"]
    assert receipt["unknown_evidence_ids"] == ["ghost"]

    restore_state = FakeWorkingSet(["e1"], [])
    receipt = restore_working_set(restore_state, ["e1", "e1", "ghost"], snapshot=lambda: {})
    assert receipt["restored_evidence_ids"] == ["e1"]
    assert receipt["not_restored_evidence_ids"] == ["ghost"]
    assert receipt["unknown_evidence_ids"] == ["ghost"]


def test_unknown_ids_get_close_archived_candidates_in_receipt():
    archived = "task-0003-ev-d7f88304c64cb41b"
    state = FakeWorkingSet([archived], [])
    receipt = restore_working_set(
        state, ["task-0003-ev-d7f88304c64cb14b"], snapshot=lambda: {}
    )  # 手误换序
    assert receipt["unknown_evidence_ids"] == ["task-0003-ev-d7f88304c64cb14b"]
    assert receipt["did_you_mean"] == {
        "task-0003-ev-d7f88304c64cb14b": [archived],
    }


def test_clean_receipt_has_no_did_you_mean_key():
    state = FakeWorkingSet(["e1"], [])
    receipt = restore_working_set(state, ["e1"], snapshot=lambda: {})
    assert "did_you_mean" not in receipt


def test_inbound_evidence_id_fields_strip_control_noise_and_dedupe():
    # 模型在超大快照里转写长 ID 会混入 \r 等噪声；字段边界归一化后按原顺序去重。
    noisy = ["task\r-0005-ev-7382e2a63f32a52a", "task-0005-ev-7382e2a63f32a52a"]
    assert ReleaseEvidence(evidence_ids=noisy, reason="r").evidence_ids == [
        "task-0005-ev-7382e2a63f32a52a"
    ]
    assert RestoreEvidence(evidence_ids=noisy, reason="r").evidence_ids == [
        "task-0005-ev-7382e2a63f32a52a"
    ]
    assert ReadEvidence(evidence_ids=noisy, reason="r").evidence_ids == [
        "task-0005-ev-7382e2a63f32a52a"
    ]
    assert ResearchDirectionComplete(
        reason="r", selected_evidence_ids=noisy
    ).selected_evidence_ids == ["task-0005-ev-7382e2a63f32a52a"]
    assert CompleteReport(
        title="测试标题", selected_evidence_ids=noisy, markdown="正文"
    ).selected_evidence_ids == ["task-0005-ev-7382e2a63f32a52a"]
    aspect = ResearchAspect(
        aspect_id="a1",
        topic="主题",
        role="支撑",
        status="covered",
        evidence_ids=noisy,
    )
    assert aspect.evidence_ids == ["task-0005-ev-7382e2a63f32a52a"]


def test_normalize_keeps_non_string_items_for_field_errors():
    assert normalize_evidence_ids(["a b", 5]) == ["ab", 5]
    assert normalize_evidence_ids("not-a-list") == "not-a-list"
