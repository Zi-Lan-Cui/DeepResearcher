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


def test_every_state_channel_model_is_allowlisted():
    """ResearchState 通道里的每个模型类型必须被 serde 扫描收录。

    白名单按包扫描收集,新增通道若落在包外(或 import 方式变化使扫描漏收),
    恢复路径会在未来某次严格化时静默失败——该测试保证问题今天暴露。
    """
    from typing import Annotated, get_args, get_origin, get_type_hints

    from pydantic import BaseModel

    from deepresearcher.state import ResearchState

    def model_types(annotation):
        origin = get_origin(annotation)
        if origin is Annotated:
            return model_types(get_args(annotation)[0])
        if origin is not None:  # list[X] | X | None 等
            for arg in get_args(annotation):
                if arg is not type(None):
                    yield from model_types(arg)
            return
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            yield annotation

    keys = project_state_types()
    for name, annotation in get_type_hints(ResearchState).items():
        for model_cls in model_types(annotation):
            if not model_cls.__module__.startswith("deepresearcher"):
                continue  # langchain 消息等第三方类型由其内建序列化承载
            assert (model_cls.__module__, model_cls.__name__) in keys, name


def test_scalar_channels_roundtrip_with_full_fidelity():
    """全 scalar 通道 dump→load 后仍是被登记模型且等值——防止绕过校验被当作合法模型。"""
    from deepresearcher.schemas import ReviewProgress, RunStatus, SupervisorProgress, WriterProgress

    serde = build_checkpointer_serde()
    samples = [
        RunStatus(phase="failed", terminal_reason="node_failed"),
        SupervisorProgress(status="incomplete", coverage_gaps=["缺来源"], is_sufficient=False),
        WriterProgress(status="exhausted", attempts=2),
        ReviewProgress(status="rejected", feedback="需要补来源"),
    ]
    for sample in samples:
        restored = serde.loads_typed(serde.dumps_typed(sample))
        assert type(restored) is type(sample)
        assert restored == sample
