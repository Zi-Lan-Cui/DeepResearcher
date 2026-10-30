"""工作台与综合稿的数据契约：入参校验、aspect 身份、证据归并、视图投影。"""

import pytest
from pydantic import ValidationError

from deepresearcher.agents.supervisor.state import (
    evidence_card,
    synthesis_snapshot,
)
from deepresearcher.schemas import (
    WORKBENCH_INTENT_ADAPTER,
    AspectIssue,
    ResearchAspect,
    ResearchSynthesis,
    SynthesisFraming,
    TaskResult,
    TaskStopReason,
    UpdateWorkbench,
    WorkbenchUpdateFraming,
)
from fakes import evidence


def test_workbench_arguments_leave_selected_derivation_to_program():
    batch = UpdateWorkbench.model_validate(
        {
            "changes": [
                {
                    "op": "upsert_aspect",
                    "aspect": {
                        "aspect_id": "core",
                        "topic": "核心",
                        "role": "主线",
                        "status": "covered",
                        "conclusion": "已有依据",
                        "evidence_ids": ["e1"],
                    },
                },
                {"op": "framing", "framing": {"thesis": "已有一项事实。"}},
            ]
        }
    )
    assert "selected_evidence_ids" not in str(batch.model_dump())
    intent = WORKBENCH_INTENT_ADAPTER.validate_python(batch.changes[1])
    assert isinstance(intent, WorkbenchUpdateFraming)
    assert intent.framing.answer_goal is None  # 省略字段=不动，由候选合成沿用现稿
    with pytest.raises(ValidationError):  # 拼错的 framing 字段名当场拒绝，不静默丢弃
        SynthesisFraming.model_validate({"unknown_field": 1})


def test_synthesis_derives_stable_evidence_union_from_aspects() -> None:
    synthesis = ResearchSynthesis.model_validate(
        {
            "revision": 1,
            "based_on_working_set_revision": 2,
            "answer_goal": "回答问题",
            "thesis": "综合结论",
            "aspects": [
                {
                    "aspect_id": "a",
                    "topic": "方面 A",
                    "role": "定义",
                    "status": "covered",
                    "evidence_ids": ["e2", "e1"],
                },
                {
                    "aspect_id": "b",
                    "topic": "方面 B",
                    "role": "对比",
                    "status": "covered",
                    "evidence_ids": ["e1", "e3"],
                },
            ],
            "selected_evidence_ids": ["legacy-extra"],
            "decision_rationale": "已形成证据链",
        }
    )

    assert synthesis.selected_evidence_ids == ["e2", "e1", "e3"]


def test_synthesis_rejects_duplicate_aspect_ids() -> None:
    payload = {
        "revision": 1,
        "based_on_working_set_revision": 1,
        "answer_goal": "回答问题",
        "thesis": "综合结论",
        "aspects": [
            {
                "aspect_id": "same",
                "topic": f"方面 {index}",
                "role": "主线",
                "status": "covered",
                "evidence_ids": [f"e{index}"],
            }
            for index in (1, 2)
        ],
        "decision_rationale": "已形成证据链",
    }

    with pytest.raises(ValueError, match="重复 aspect_id"):
        ResearchSynthesis.model_validate(payload)


def test_aspect_grounding_requires_evidence_or_explicit_gap() -> None:
    """状态层只校验内部一致性：covered 必须挂证据，uncovered 不得挂证据，缺口必须写明。"""
    base = {"aspect_id": "a", "topic": "主题", "role": "主线"}
    with pytest.raises(ValueError, match="covered 研究方面必须绑定"):
        ResearchAspect(**base, status="covered", evidence_ids=[])
    with pytest.raises(ValueError, match="uncovered 研究方面不能绑定"):
        ResearchAspect(
            **base, status="uncovered", evidence_ids=["e1"], gaps=[AspectIssue(text="缺来源")]
        )
    with pytest.raises(ValueError, match="必须登记至少一条 gap"):
        ResearchAspect(**base, status="partial", evidence_ids=["e1"])


def test_supervisor_views_preserve_metadata_without_exposing_quote() -> None:
    item = evidence("metadata", task_id="r1-1")
    item.source_url = "https://docs.example.com/report"
    item.source_title = "来源标题"
    item.published_at = "2026-09-01"
    card = evidence_card(item)

    assert card["source_title"] == "来源标题"
    assert card["source_domain"] == "docs.example.com"
    assert card["published_at"] == "2026-09-01"
    assert "quote" not in card

    synthesis = ResearchSynthesis.model_validate(
        {
            "revision": 1,
            "based_on_working_set_revision": 1,
            "answer_goal": "回答问题",
            "thesis": "综合结论",
            "aspects": [
                {
                    "aspect_id": "core",
                    "topic": "核心",
                    "role": "结论主线",
                    "required": False,
                    "status": "covered",
                    "evidence_ids": [item.evidence_id],
                }
            ],
            "decision_rationale": "决策理由",
        }
    )
    snapshot = synthesis_snapshot(synthesis)

    assert snapshot["synthesis_revision"] == 1
    card = snapshot["research_synthesis"]
    assert card["aspects"][0]["role"] == "结论主线"
    assert card["aspects"][0]["required"] is False
    assert card["decision_rationale"] == "决策理由"


def test_working_set_snapshot_resolves_task_titles_both_forms() -> None:
    """方向短题从 task_results 回查（BaseModel 与检查点 dict 双形态）；无短题回退长契约截断。"""
    from deepresearcher.agents.supervisor.state import workbench_view as working_set_snapshot

    titled = evidence("titled", task_id="task-0001")
    titled.research_direction = "梳理软件服务一般特征的权威定义与出处边界并逐条标注出处要求"
    dict_titled = evidence("dict", task_id="task-0002")
    dict_titled.research_direction = "长契约文本不该出现在带短题的卡上"
    fallback = evidence("fallback", task_id="task-0004")  # 无 task_result：短契约原样显示
    fallback.research_direction = "短方向"
    long_without_title = evidence("long", task_id="task-0003")
    long_without_title.research_direction = "甲" * 60

    class _SnapshotState:
        def __init__(self) -> None:
            self.evidences = [titled, dict_titled, fallback, long_without_title]
            self.active_evidence_ids = {titled.evidence_id}
            self.active_evidence_limit = 6
            self.working_set_revision = 1
            self.pending_tasks: dict = {}
            self.plan: list = []
            self.provider_exhausted = False
            self.research_synthesis = None
            self.max_turns = 34
            self.current_round = 1
            self.task_results = [
                TaskResult(
                    task_id="task-0001",
                    round=1,
                    question=titled.research_direction,
                    research_direction=titled.research_direction,
                    display_title="特征体系",
                    execution_status="completed",
                    coverage_status="sufficient",
                    evidence_count=1,
                    source_count=1,
                    stop_reason=TaskStopReason.COMPLETE,
                ),
                {"task_id": "task-0002", "display_title": "世代语境"},
            ]

        def active_evidences(self) -> list:
            return [item for item in self.evidences if item.evidence_id in self.active_evidence_ids]

    snapshot = working_set_snapshot(_SnapshotState())

    assert snapshot["active_evidence"][0]["task_title"] == "特征体系"
    reserve_cards = {card["task_title"] for card in snapshot["reserve_evidence"]}
    assert "世代语境" in reserve_cards  # dict 恢复形态
    assert "甲" * 29 + "…" in reserve_cards  # 无短题的长契约截断到 30 字符
    assert "短方向" in reserve_cards
