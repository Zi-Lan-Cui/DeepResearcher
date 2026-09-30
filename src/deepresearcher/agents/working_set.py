"""supervisor 与 researcher 共享的 Evidence 工作集释放/恢复原语。

两侧的工作集操作同源(archive=全量档案、active=工作集、reserve=档案−工作集),
回执此前在两边 tools.py 各维护一份并已漂移:去重先后不同、supervisor 的 restore
缺 unknown_evidence_ids。本模块收拢为一份实现,行为取严的一侧——请求 id 先按
顺序去重再进回执;restore 与 release 同为三键形状。
快照回执因两侧卡片形状不同而由调用方以回调传入,并在状态变更后求值。
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

from deepresearcher.evidence.models import Evidence


class WorkingSetState(Protocol):
    """共享原语依赖的最小工作集接口;两侧 LoopState 结构性满足,不做 isinstance 校验。"""

    evidences: list[Evidence]
    active_evidence_ids: set[str]

    def release_evidence(self, evidence_ids: list[str]) -> list[str]: ...

    def restore_evidence(self, evidence_ids: list[str]) -> list[str]: ...


def release_working_set(
    loop_state: WorkingSetState,
    evidence_ids: Sequence[str],
    *,
    snapshot: Callable[[], Mapping[str, object]],
) -> dict[str, object]:
    """从工作集释放(不删档案),返回统一三键 + 变更后快照的回执。"""
    requested = list(dict.fromkeys(evidence_ids))
    released = loop_state.release_evidence(requested)
    archive_ids = {item.evidence_id for item in loop_state.evidences}
    return {
        "released_evidence_ids": released,
        # 重复释放同一 id ≠ 编造:档案在而工作集无,单列一键;
        # unknown 只留给真不在档案的 id——与 Restore 的键形状对齐。
        "not_in_working_set_ids": [
            item for item in requested if item in archive_ids and item not in released
        ],
        "unknown_evidence_ids": [item for item in requested if item not in archive_ids],
        **snapshot(),
    }


def restore_working_set(
    loop_state: WorkingSetState,
    evidence_ids: Sequence[str],
    *,
    snapshot: Callable[[], Mapping[str, object]],
) -> dict[str, object]:
    """从档案恢复进工作集(受活跃上限约束),返回统一三键 + 变更后快照的回执。"""
    requested = list(dict.fromkeys(evidence_ids))
    restored = loop_state.restore_evidence(requested)
    archive_ids = {item.evidence_id for item in loop_state.evidences}
    return {
        "restored_evidence_ids": restored,
        "not_restored_evidence_ids": [item for item in requested if item not in restored],
        "unknown_evidence_ids": [item for item in requested if item not in archive_ids],
        **snapshot(),
    }
