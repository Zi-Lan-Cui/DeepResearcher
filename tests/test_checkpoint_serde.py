from deepresearcher.schemas import ResearchDirectionResult
from deepresearcher.service.checkpoint_serde import build_checkpointer_serde, project_state_types


def test_allowlist_covers_our_state_types():
    keys = project_state_types()
    # 触发过告警的三个类型都必须在允许清单里（否则恢复在严格模式下会炸）。
    assert ("deepresearcher.schemas.sections", "ResearchDirectionResult") in keys


def test_state_model_roundtrips_through_serde():
    serde = build_checkpointer_serde()
    model = ResearchDirectionResult(
        task_id="r1-1",
        round=1,
        question="q",
        research_direction="q",
        execution_status="completed",
        coverage_status="sufficient",
        evidence_count=1,
        source_count=1,
        conclusion="done",
        stop_reason="complete",
    )
    dumped = serde.dumps_typed(model)
    restored = serde.loads_typed(dumped)
    assert restored == model
