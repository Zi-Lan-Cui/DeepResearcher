"""telemetry_bridge 的降级路径与属性清洗单测。

真实 span 树(含 provider 断言)在 test_observability_opentelemetry.py,
因为全局 TracerProvider 一经设置整个测试进程共享,只能由单一模块持有。
"""

from deepresearcher.observability.tracing import telemetry_bridge
from deepresearcher.observability.tracing.telemetry_bridge import (
    BridgeSpan,
    _sanitized_attributes,
    begin_span,
    override_enabled,
)


def test_disabled_bridge_yields_null_handle_and_finish_is_noop():
    with override_enabled(False):
        with begin_span("research_run", kind="trace", attributes={"k": "v"}) as handle:
            assert handle.otel_trace_id is None
            assert handle.otel_span_id is None
            handle.finish("completed")
            handle.finish("failed", "重复定终态不报错")


def test_begin_span_runs_body_without_raising():
    # 全局 provider 状态取决于测试文件执行顺序(设置是一次性的),两种形态都要能走通:
    # 未配置时 OTel 给 NonRecordingSpan(句柄无 id),已配置时给合法 hex id。
    with begin_span("node_body") as handle:
        assert handle.otel_trace_id is None or len(handle.otel_trace_id) == 32


def test_body_exception_propagates_unchanged_when_disabled():
    class ProbeError(Exception):
        pass

    try:
        with override_enabled(False):
            with begin_span("x"):
                raise ProbeError("体内异常必须原样重抛")
    except ProbeError as exc:
        assert str(exc) == "体内异常必须原样重抛"
    else:
        raise AssertionError("begin_span 吞掉了体内异常")


def test_sanitized_attributes_keeps_scalars_stringifies_rest():
    class _Named:
        def __str__(self) -> str:
            return "字符串化"

    cleaned = _sanitized_attributes(
        {"text": "abc", "count": 3, "ratio": 0.5, "flag": True, "missing": None, "object": _Named()}
    )
    assert cleaned == {"text": "abc", "count": 3, "ratio": 0.5, "flag": True, "object": "字符串化"}


def test_null_bridge_span_singleton_is_finish_idempotent():
    handle = BridgeSpan(None, None, None)
    handle.finish("completed")
    handle.finish("failed", "无 span 时全部为空操作")
    assert handle.otel_trace_id is None


def test_instrumentation_scope_and_kind_maps_are_consistent():
    # kind 词表由 recorder 的调用面决定;映射缺项会让新 kind 静默落到 INTERNAL。
    assert set(telemetry_bridge._SPAN_KIND_MAP) >= {"trace", "node", "tool"}
    assert telemetry_bridge._INSTRUMENTATION_SCOPE == "deepresearcher.tracing"
