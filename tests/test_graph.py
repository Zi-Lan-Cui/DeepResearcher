import asyncio
from pathlib import Path

import pytest

from deepsearch_agent.agents.writer import ReportWriter
from deepsearch_agent.config import (
    AgentConfig,
    AppConfig,
    LLMConfig,
    ObservabilityConfig,
    SearchConfig,
    Settings,
)
from deepsearch_agent.llm import LLMConfigurationError
from deepsearch_agent.observability.events.models import NodeEvent
from deepsearch_agent.orchestration import graph, nodes
from deepsearch_agent.orchestration.execution_boundary import execute_node
from deepsearch_agent.orchestration.graph import _route_after_reflection, build_graph
from deepsearch_agent.schemas import (
    Citation,
    ClarificationDecision,
    ParagraphBinding,
    ResearchProgress,
    ReviewIssue,
    RouteDecision,
    RunError,
    RunLifecycle,
    WriterProgress,
)
from deepsearch_agent.tools.errors import ToolRequestError


class TextLLM:
    async def ainvoke_text(self, _messages, **_kwargs):
        return type("Response", (), {"content": "这是显式注入的即时回答。"})()

    def bind_tools(self, _tools, **_kwargs):
        return self


class GraphLLM:
    """只测试图装配和路由时使用的最小工具绑定假模型。"""

    def bind_tools(self, _tools, tool_choice="any", **_kwargs):
        return self


def _graph_settings(tmp_path: Path) -> Settings:
    return Settings(
        llm=LLMConfig(),
        agent=AgentConfig(),
        search=SearchConfig(tavily_api_key="test-key"),
        app=AppConfig(),
        observability=ObservabilityConfig(log_dir=tmp_path),
    )


async def _deep_research_route(_state, _llm):
    return {"route": "deep_research", "route_reason": "测试"}


async def _clarify_success(state, _llm):
    query = state["query"]
    return {"clarified_query": query, "research_brief": "测试研究方向"}


class _FailingSupervisor:
    def __init__(self, *_args, **_kwargs):
        pass

    async def run(self, _state):
        raise RuntimeError("supervisor boom")


class _CompleteSupervisor:
    def __init__(self, *_args, **_kwargs):
        pass

    async def run(self, _state):
        return {
            "run": RunLifecycle(phase="writing"),
            "research": ResearchProgress(status="completed", current_round=1, is_sufficient=True, generation_mode="full"),
            "supervisor_next": "writer",
            "evidences": [],
        }


class _ExhaustedSupervisor:
    def __init__(self, *_args, **_kwargs):
        pass

    async def run(self, _state):
        return {
            "run": RunLifecycle(phase="rendering", terminal_reason="research_budget_exhausted"),
            "research": ResearchProgress(status="incomplete", current_round=1),
            "answer_mode": "research_incomplete",
            "supervisor_next": "render_final_report",
            "report": "研究轮次已耗尽。",
        }


class _FailingWriter:
    def __init__(self, *_args, **_kwargs):
        pass

    async def run(self, _state):
        raise RuntimeError("writer boom")


class _ReadyWriter:
    def __init__(self, *_args, **_kwargs):
        pass

    async def run(self, _state):
        citation = Citation(id="e1", claim="测试事实", quote="测试事实。")
        return {
            "run": RunLifecycle(phase="reviewing"),
            "writer": WriterProgress(status="completed"),
            "answer_mode": "deep_research",
            "report_draft": "测试事实。 [[cite:e1]]",
            "citations": [citation],
            "paragraph_bindings": [
                ParagraphBinding(text="测试事实。", kind="evidence", evidence_ids=["e1"])
            ],
        }


def test_quick_answer_route_produces_uncited_answer():
    state = asyncio.run(nodes.quick_answer({"query": "什么是向量数据库"}, TextLLM()))
    writer = ReportWriter(
        TextLLM(),
        AgentConfig(),
        render_incomplete=lambda _state: "incomplete",
    )
    result = asyncio.run(writer.run(state))
    assert result["answer_mode"] == "quick_answer"
    assert "未进行联网检索或来源核验" in result["report"]


def test_router_delegates_research_classification_to_llm():
    class RoutingLLM:
        async def ainvoke_structured(self, _schema, _messages, **_kwargs):
            return RouteDecision(route="quick_answer", reason="模型判断为单一问题")

    result = asyncio.run(nodes.router({"query": "哪些 galgame 具有广泛的影响力"}, RoutingLLM()))
    assert result["route"] == "quick_answer"
    assert result["route_reason"] == "模型判断为单一问题"


def test_router_model_failure_fails_closed_to_deep_research():
    class FailingLLM:
        async def ainvoke_structured(self, *_args, **_kwargs):
            raise TimeoutError("offline")

    result = asyncio.run(nodes.router({"query": "单一事实问题"}, FailingLLM()))
    assert result["route"] == "deep_research"
    assert "调用失败" in result["route_reason"]


def test_top_level_node_failure_becomes_renderable_run_error():
    async def broken(_state):
        raise RuntimeError("boom")

    result = asyncio.run(execute_node({"query": "测试"}, stage="broken", node=broken))

    assert result["run"].phase == "failed"
    assert result["run"].terminal_reason == "node_failed"
    assert result["run"].error.stage == "broken"
    assert "boom" in result["report"]


def test_state_invariant_violation_becomes_run_error():
    async def contradictory_node(_state):
        return {"run": RunLifecycle(phase="running", terminal_reason="already done")}

    result = asyncio.run(
        execute_node(
            {"query": "测试"},
            stage="contradictory",
            node=contradictory_node,
        )
    )

    assert result["run"].phase == "failed"
    assert result["run"].error.stage == "contradictory"
    assert "phase" in result["run"].error.message


def test_failed_state_without_error_is_rejected():
    async def incomplete_failure(_state):
        return {"run": RunLifecycle(phase="failed", terminal_reason="失败")}

    result = asyncio.run(
        execute_node(
            {"query": "测试"},
            stage="invalid_failure",
            node=incomplete_failure,
        )
    )

    assert result["run"].phase == "failed"
    assert result["run"].error.stage == "invalid_failure"
    assert "run" in result["run"].error.message


def test_typed_agent_error_preserves_code_and_retryability():
    async def transient_tool_failure(_state):
        raise ToolRequestError("上游暂时不可用")

    result = asyncio.run(
        execute_node(
            {"query": "测试"},
            stage="search",
            node=transient_tool_failure,
        )
    )

    assert result["run"].error.code == "tool_request"
    assert result["run"].error.retryable is True


def test_execution_boundary_restores_checkpoint_models_before_node():
    async def inspect_state(state):
        assert isinstance(state["run"].error, RunError)
        assert isinstance(state["review"].issues[0], ReviewIssue)
        assert isinstance(state["citations"][0], Citation)
        assert isinstance(state["node_events"][0], NodeEvent)
        return {"run": RunLifecycle(phase="routing")}

    result = asyncio.run(
        execute_node(
            {
                "run": {"phase": "failed", "error": {"stage": "writer", "code": "node_failed", "message": "失败"}},
                "review": {"issues": [{"severity": "warning", "reason": "可改进"}]},
                "citations": [{"id": "e1"}],
                "node_events": [
                    {
                        "event_id": "evt-1",
                        "event_type": "node_completed",
                        "timestamp": "2026-01-01T00:00:00+00:00",
                        "node": "writer",
                        "status": "completed",
                    }
                ],
            },
            stage="inspect",
            node=inspect_state,
        )
    )
    assert result["run"].phase == "routing"


def test_graph_requires_llm_at_assembly(monkeypatch):
    def missing_llm(_settings):
        raise LLMConfigurationError("missing")

    monkeypatch.setattr(graph, "build_llm", missing_llm)
    with pytest.raises(LLMConfigurationError, match="missing"):
        build_graph()


def test_compiled_graph_clarify_failure_renders_failure_report(tmp_path, monkeypatch):
    monkeypatch.setattr(graph.nodes, "router", _deep_research_route)
    monkeypatch.setattr(graph, "ResearchSupervisor", _FailingSupervisor)

    async def fail_clarify(_state, _llm):
        raise RuntimeError("clarify boom")

    monkeypatch.setattr(graph.nodes, "clarify", fail_clarify)

    result = asyncio.run(
        build_graph(_graph_settings(tmp_path), llm=GraphLLM()).ainvoke({"query": "测试"})
    )

    assert result["run"].phase == "failed"
    assert result["run"].error.stage == "clarify"
    assert "clarify boom" in result["report"]


def test_compiled_graph_router_failure_fails_closed_to_deep_research(tmp_path, monkeypatch):
    async def fail_structured(*_args, **_kwargs):
        raise TimeoutError("router offline")

    monkeypatch.setattr(graph.nodes, "ainvoke_structured", fail_structured)
    monkeypatch.setattr(graph.nodes, "clarify", _clarify_success)
    monkeypatch.setattr(graph, "ResearchSupervisor", _FailingSupervisor)

    result = asyncio.run(
        build_graph(_graph_settings(tmp_path), llm=GraphLLM()).ainvoke({"query": "测试"})
    )

    assert result["route"] == "deep_research"
    assert result["run"].error.stage == "supervisor"


def test_compiled_graph_supervisor_failure_skips_writer(tmp_path, monkeypatch):
    monkeypatch.setattr(graph.nodes, "router", _deep_research_route)
    monkeypatch.setattr(graph.nodes, "clarify", _clarify_success)
    monkeypatch.setattr(graph, "ResearchSupervisor", _FailingSupervisor)

    result = asyncio.run(
        build_graph(_graph_settings(tmp_path), llm=GraphLLM()).ainvoke({"query": "测试"})
    )

    assert result["run"].error.stage == "supervisor"
    assert result.get("writer") is None
    assert "supervisor boom" in result["report"]


def test_compiled_graph_writer_failure_renders_failure_report(tmp_path, monkeypatch):
    monkeypatch.setattr(graph.nodes, "router", _deep_research_route)
    monkeypatch.setattr(graph.nodes, "clarify", _clarify_success)
    monkeypatch.setattr(graph, "ResearchSupervisor", _CompleteSupervisor)
    monkeypatch.setattr(graph, "ReportWriter", _FailingWriter)

    result = asyncio.run(
        build_graph(_graph_settings(tmp_path), llm=GraphLLM()).ainvoke({"query": "测试"})
    )

    assert result["run"].error.stage == "writer"
    assert "writer boom" in result["report"]


def test_compiled_graph_reflection_failure_renders_failure_report(tmp_path, monkeypatch):
    monkeypatch.setattr(graph.nodes, "router", _deep_research_route)
    monkeypatch.setattr(graph.nodes, "clarify", _clarify_success)
    monkeypatch.setattr(graph, "ResearchSupervisor", _CompleteSupervisor)
    monkeypatch.setattr(graph, "ReportWriter", _ReadyWriter)

    async def fail_reflection(_state, _llm):
        raise RuntimeError("reflection boom")

    monkeypatch.setattr(graph.nodes, "reflection", fail_reflection)

    result = asyncio.run(
        build_graph(_graph_settings(tmp_path), llm=GraphLLM()).ainvoke({"query": "测试"})
    )

    assert result["run"].error.stage == "reflection"
    assert "reflection boom" in result["report"]


def test_compiled_graph_research_exhaustion_renders_incomplete_report(tmp_path, monkeypatch):
    monkeypatch.setattr(graph.nodes, "router", _deep_research_route)
    monkeypatch.setattr(graph.nodes, "clarify", _clarify_success)
    monkeypatch.setattr(graph, "ResearchSupervisor", _ExhaustedSupervisor)

    result = asyncio.run(
        build_graph(_graph_settings(tmp_path), llm=GraphLLM()).ainvoke({"query": "测试"})
    )

    assert result["run"].phase == "completed"
    assert result["run"].terminal_reason == "research_incomplete"
    assert "research_budget_exhausted" in result["report"]


def test_compiled_graph_preserves_cancellation(tmp_path, monkeypatch):
    monkeypatch.setattr(graph.nodes, "router", _deep_research_route)
    monkeypatch.setattr(graph, "ResearchSupervisor", _FailingSupervisor)

    async def cancel_clarify(_state, _llm):
        raise asyncio.CancelledError()

    monkeypatch.setattr(graph.nodes, "clarify", cancel_clarify)

    # LangGraph 会把节点主动抛出的 CancelledError 包装成自己的取消异常；
    # 关键契约是它不会被执行边界转换成 RunError 并渲染成失败报告。
    with pytest.raises(BaseException) as caught:
        asyncio.run(build_graph(_graph_settings(tmp_path), llm=GraphLLM()).ainvoke({"query": "测试"}))
    assert type(caught.value).__name__ != "RunError"


def test_approved_reflection_bypasses_supervisor_and_renders_final_report():
    assert _route_after_reflection({"review": {"status": "approved"}}) == "render_final_report"
    assert _route_after_reflection({"review": {"status": "rejected"}}) == "supervisor"


def test_clarify_preserves_open_question_and_only_adds_research_brief(monkeypatch):
    async def clarified(llm, schema, messages, **kwargs):
        assert schema is ClarificationDecision
        return ClarificationDecision(
            intent_summary="分析理想化世界在 Galgame 与文学中的叙事功能和意义。",
            research_focus=["现实复杂性的取舍", "情感体验与主题表达", "理想化的边界"],
        )

    monkeypatch.setattr(nodes, "ainvoke_structured", clarified)
    query = "如何看待galgame或者文学作品中的理想简化的世界"
    result = asyncio.run(nodes.clarify({"query": query}, object()))

    assert result["clarified_query"] == query
    assert "叙事功能" in result["research_brief"]
    assert result.get("answer_mode") is None


def test_clarify_stops_only_for_material_user_choice(monkeypatch):
    async def needs_input(llm, schema, messages, **kwargs):
        return ClarificationDecision(
            needs_user_input=True,
            intent_summary="用户需要选择分析对象。",
            clarification_question="你希望讨论具体作品，还是只讨论一般叙事机制？",
        )

    monkeypatch.setattr(nodes, "ainvoke_structured", needs_input)
    result = asyncio.run(nodes.clarify({"query": "分析它"}, object()))

    assert result["answer_mode"] == "clarification_needed"
    assert "具体作品" in result["report"]
