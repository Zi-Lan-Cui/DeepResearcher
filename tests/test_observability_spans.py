"""spans 模块的建 span 语义:creation-time 标签注入、属性清洗与词表。

统一 span 树的端到端断言在 test_observability_opentelemetry.py;
账本记录形状在 test_observability.py。
"""

from deepresearcher.observability.tracing import spans
from deepresearcher.observability.tracing.context import bind_context


def test_label_attributes_snapshot_business_labels():
    with bind_context(run_id="run-1", session_id="sess-1", node_id="supervisor"):
        attributes = spans._label_attributes("node")
    assert attributes == {
        "deepresearcher.kind": "node",
        "deepresearcher.run_id": "run-1",
        "deepresearcher.session_id": "sess-1",
        "deepresearcher.node_id": "supervisor",
    }


def test_label_attributes_without_binding_yields_none_labels():
    attributes = spans._label_attributes("trace")
    assert attributes["deepresearcher.run_id"] is None
    assert attributes["deepresearcher.node_id"] is None


def test_sanitized_keeps_scalars_stringifies_rest():
    class _Named:
        def __str__(self) -> str:
            return "字符串化"

    cleaned = spans._sanitized(
        {"text": "abc", "count": 3, "ratio": 0.5, "flag": True, "missing": None, "object": _Named()}
    )
    assert cleaned == {"text": "abc", "count": 3, "ratio": 0.5, "flag": True, "object": "字符串化"}


def test_kind_maps_cover_the_call_site_vocabulary():
    # kind 词表由 instrumentation/tools 的调用面决定;缺项会静默落到 INTERNAL。
    assert set(spans._SPAN_KIND_MAP) >= {"trace", "node", "tool"}
    assert spans.INSTRUMENTATION_SCOPE == "deepresearcher.tracing"


def test_span_body_exception_propagates_after_marking():
    class ProbeError(Exception):
        pass

    try:
        with spans.span("boom"):
            raise ProbeError("体内异常必须原样重抛")
    except ProbeError as exc:
        assert str(exc) == "体内异常必须原样重抛"
    else:
        raise AssertionError("span 吞掉了体内异常")


def test_unregistered_run_writes_no_ledger_and_raises_nothing():
    # 账本路由查无 sink 时静默丢弃:span 照常执行,不产生任何写出错误。
    with spans.span("orphan"):
        pass
