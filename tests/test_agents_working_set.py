"""共享 working-set 原语的契约测试:三键回执形状、幂等去重、快照后置于变更。"""

from deepresearcher.agents.working_set import release_working_set, restore_working_set
from deepresearcher.evidence.models import Evidence


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
    """最小 LoopState 替身:档案与活跃集、两转移方法。"""

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
    # 快照必须在释放之后求值:活跃数应已减 1。
    assert receipt["active_evidence_count"] == 1


def test_duplicate_and_unknown_ids_classified_consistently_on_both_ops():
    release_state = FakeWorkingSet(["e1"], ["e1"])
    receipt = release_working_set(
        release_state, ["e1", "e1", "ghost"], snapshot=lambda: {}
    )
    # 请求按序去重:重复 id 不在任何键里出现两次;ghost 不在档案 → unknown。
    assert receipt["released_evidence_ids"] == ["e1"]
    assert receipt["unknown_evidence_ids"] == ["ghost"]

    restore_state = FakeWorkingSet(["e1"], [])
    receipt = restore_working_set(
        restore_state, ["e1", "e1", "ghost"], snapshot=lambda: {}
    )
    assert receipt["restored_evidence_ids"] == ["e1"]
    assert receipt["not_restored_evidence_ids"] == ["ghost"]
    assert receipt["unknown_evidence_ids"] == ["ghost"]
