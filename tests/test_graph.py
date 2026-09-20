import asyncio
from pathlib import Path

import pytest
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph

from deepresearcher.agents.clarifier.state import ClarifierAgentState
from deepresearcher.agents.writer import ReportWriter
from deepresearcher.config import (
    AgentConfig,
    AppConfig,
    LLMConfig,
    ObservabilityConfig,
    SearchConfig,
    Settings,
)
from deepresearcher.llm import LLMConfigurationError
from deepresearcher.observability.events.models import NodeEvent
from deepresearcher.orchestration import graph, nodes
from deepresearcher.orchestration.execution_boundary import execute_node
from deepresearcher.orchestration.graph import build_graph
from deepresearcher.routing import (
    NodeName,
    route_after_reflection,
    route_after_supervisor,
    route_after_writer,
)
from deepresearcher.schemas import (
    Citation,
    ParagraphBinding,
    ReviewIssue,
    RouteDecision,
    RunError,
    RunStatus,
    SupervisorProgress,
    WriterProgress,
)
from deepresearcher.service.usage import UsageBudgetExceeded
from deepresearcher.state import validate_state_invariants
from deepresearcher.tools.errors import ToolRequestError


class TextLLM:
    async def ainvoke(self, _messages, **_kwargs):
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


class _FakeClarifier:
    async def run(self, state):
        return await self.graph.ainvoke(state)


class _CompleteClarifier(_FakeClarifier):
    def __init__(self, *_args, **_kwargs):
        self.graph = _fake_clarifier_graph(_clarify_success)


class _FailingClarifier(_FakeClarifier):
    def __init__(self, *_args, **_kwargs):
        async def fail(_state, _llm):
            raise RuntimeError("clarify boom")

        self.graph = _fake_clarifier_graph(fail)


class _CancellingClarifier(_FakeClarifier):
    def __init__(self, *_args, **_kwargs):
        async def cancel(_state, _llm):
            raise asyncio.CancelledError()

        self.graph = _fake_clarifier_graph(cancel)


def _fake_clarifier_graph(node):
    builder = StateGraph(ClarifierAgentState)

    async def invoke(state):
        result = await node(state, None)
        return {
            "intent_summary": result.get("research_brief", "测试研究方向"),
            "clarification_completed": True,
        }

    builder.add_node("clarify", invoke)
    builder.add_edge(START, "clarify")
    builder.add_edge("clarify", END)
    return builder.compile()


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
            "run": RunStatus(phase="writing"),
            "supervisor": SupervisorProgress(
                status="completed", current_round=1, is_sufficient=True, generation_mode="full"
            ),
            "supervisor_next": "writer",
            "evidences": [],
        }


class _ExhaustedSupervisor:
    def __init__(self, *_args, **_kwargs):
        pass

    async def run(self, _state):
        return {
            "run": RunStatus(phase="rendering", terminal_reason="research_budget_exhausted"),
            "supervisor": SupervisorProgress(status="incomplete", current_round=1),
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
            "run": RunStatus(phase="reviewing"),
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
    assert result["report"] == "这是显式注入的即时回答。"
    assert "什么是向量数据库" not in result["report"]


def test_router_delegates_research_classification_to_llm():
    class RoutingLLM:
        def with_structured_output(self, _schema, **_kwargs):
            class _Decision:
                async def ainvoke(self, _messages, **_kw):
                    return RouteDecision(route="quick_answer", reason="模型判断为单一问题")

            return _Decision()

    result = asyncio.run(nodes.router({"query": "哪些 galgame 具有广泛的影响力"}, RoutingLLM()))
    assert result["route"] == "quick_answer"
    assert result["route_reason"] == "模型判断为单一问题"


def test_router_model_failure_fails_closed_to_deep_research():
    class FailingLLM:
        def with_structured_output(self, _schema, **_kwargs):
            class _Boom:
                async def ainvoke(self, _messages, **_kw):
                    raise TimeoutError("offline")

            return _Boom()

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
    assert "执行失败，请稍后重试" in result["report"]
    assert "boom" not in result["report"]
    # 边界的失败产物自身必须通过与节点产物相同的不变量校验（回归锁）。
    validate_state_invariants({}, result)


def test_usage_budget_exhaustion_degrades_to_incomplete_report():
    async def over_budget(_state):
        raise UsageBudgetExceeded("run_token_budget_exhausted")

    result = asyncio.run(execute_node({"query": "测试"}, stage="writer", node=over_budget))

    assert result["run"].phase == "failed"
    assert result["run"].terminal_reason == "budget_exhausted"
    assert result["run"].error.code == "budget_exhausted"
    assert "已达用量上限" in result["report"]
    assert "run_token_budget_exhausted" not in result["report"]


def test_execution_boundary_never_converts_graph_interrupt_to_failure():
    async def paused(_state):
        raise GraphInterrupt()

    with pytest.raises(GraphInterrupt):
        asyncio.run(execute_node({"query": "测试"}, stage="clarify", node=paused))


def test_state_invariant_violation_becomes_run_error():
    async def contradictory_node(_state):
        return {"run": RunStatus(phase="running", terminal_reason="already done")}

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
        return {"run": RunStatus(phase="failed", terminal_reason="失败")}

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
        return {"run": RunStatus(phase="routing")}

    result = asyncio.run(
        execute_node(
            {
                "run": {
                    "phase": "failed",
                    "error": {"stage": "writer", "code": "node_failed", "message": "失败"},
                },
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

    monkeypatch.setattr(graph, "Clarifier", _FailingClarifier)

    result = asyncio.run(
        build_graph(_graph_settings(tmp_path), llm=GraphLLM()).ainvoke({"query": "测试"})
    )

    assert result["run"].phase == "failed"
    assert result["run"].error.stage == "clarify"
    assert "执行失败，请稍后重试" in result["report"]
    assert "clarify boom" not in result["report"]


def test_compiled_graph_router_failure_fails_closed_to_deep_research(tmp_path, monkeypatch):
    async def fail_structured(*_args, **_kwargs):
        raise TimeoutError("router offline")

    monkeypatch.setattr(graph.nodes, "ainvoke_structured", fail_structured)
    monkeypatch.setattr(graph, "Clarifier", _CompleteClarifier)
    monkeypatch.setattr(graph, "ResearchSupervisor", _FailingSupervisor)

    result = asyncio.run(
        build_graph(_graph_settings(tmp_path), llm=GraphLLM()).ainvoke({"query": "测试"})
    )

    assert result["route"] == "deep_research"
    assert result["run"].error.stage == "supervisor"


def test_compiled_graph_supervisor_failure_skips_writer(tmp_path, monkeypatch):
    monkeypatch.setattr(graph.nodes, "router", _deep_research_route)
    monkeypatch.setattr(graph, "Clarifier", _CompleteClarifier)
    monkeypatch.setattr(graph, "ResearchSupervisor", _FailingSupervisor)

    result = asyncio.run(
        build_graph(_graph_settings(tmp_path), llm=GraphLLM()).ainvoke({"query": "测试"})
    )

    assert result["run"].error.stage == "supervisor"
    assert result.get("writer") is None
    assert "执行失败，请稍后重试" in result["report"]
    assert "supervisor boom" not in result["report"]


def test_compiled_graph_writer_failure_renders_failure_report(tmp_path, monkeypatch):
    monkeypatch.setattr(graph.nodes, "router", _deep_research_route)
    monkeypatch.setattr(graph, "Clarifier", _CompleteClarifier)
    monkeypatch.setattr(graph, "ResearchSupervisor", _CompleteSupervisor)
    monkeypatch.setattr(graph, "ReportWriter", _FailingWriter)

    result = asyncio.run(
        build_graph(_graph_settings(tmp_path), llm=GraphLLM()).ainvoke({"query": "测试"})
    )

    assert result["run"].error.stage == "writer"
    assert "执行失败，请稍后重试" in result["report"]
    assert "writer boom" not in result["report"]


def test_compiled_graph_reflection_failure_renders_failure_report(tmp_path, monkeypatch):
    monkeypatch.setattr(graph.nodes, "router", _deep_research_route)
    monkeypatch.setattr(graph, "Clarifier", _CompleteClarifier)
    monkeypatch.setattr(graph, "ResearchSupervisor", _CompleteSupervisor)
    monkeypatch.setattr(graph, "ReportWriter", _ReadyWriter)

    async def fail_reflection(_state, _llm):
        raise RuntimeError("reflection boom")

    monkeypatch.setattr(graph.nodes, "reflection", fail_reflection)

    result = asyncio.run(
        build_graph(_graph_settings(tmp_path), llm=GraphLLM()).ainvoke({"query": "测试"})
    )

    assert result["run"].error.stage == "reflection"
    assert "执行失败，请稍后重试" in result["report"]
    assert "reflection boom" not in result["report"]


def test_compiled_graph_research_exhaustion_renders_incomplete_report(tmp_path, monkeypatch):
    monkeypatch.setattr(graph.nodes, "router", _deep_research_route)
    monkeypatch.setattr(graph, "Clarifier", _CompleteClarifier)
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

    monkeypatch.setattr(graph, "Clarifier", _CancellingClarifier)

    # LangGraph 会把节点主动抛出的 CancelledError 包装成自己的取消异常；
    # 关键契约是它不会被执行边界转换成 RunError 并渲染成失败报告。
    with pytest.raises(BaseException) as caught:
        asyncio.run(
            build_graph(_graph_settings(tmp_path), llm=GraphLLM()).ainvoke({"query": "测试"})
        )
    assert type(caught.value).__name__ != "RunError"


def test_approved_reflection_bypasses_supervisor_and_renders_final_report():
    assert (
        route_after_reflection({"review": {"status": "approved"}}) == NodeName.RENDER_FINAL_REPORT
    )
    assert route_after_reflection({"review": {"status": "rejected"}}) == NodeName.SUPERVISOR


def test_terminal_phase_trunk_overrides_every_business_handoff():
    """主干规则：节点声明 rendering/failed 后，无论业务字段写的是什么去向都收束到渲染。"""
    exhausted_writer = {
        "run": {"phase": "rendering"},
        "writer": {"status": "exhausted"},
    }
    assert route_after_writer(exhausted_writer) == NodeName.RENDER_FINAL_REPORT
    assert route_after_writer(
        {"run": {"phase": "reviewing"}, "writer": {"status": "completed"}}
    ) == (NodeName.REFLECTION)
    # 审阅还在等回流，但 Supervisor 宣告终止 → 仍然去渲染
    assert (
        route_after_reflection({"run": {"phase": "rendering"}, "review": {"status": "rejected"}})
        == NodeName.RENDER_FINAL_REPORT
    )
    # supervisor_next 的业务交接在正常态生效
    assert route_after_supervisor({"supervisor_next": NodeName.WRITER}) == NodeName.WRITER
    assert (
        route_after_supervisor({"run": {"phase": "failed"}, "supervisor_next": NodeName.WRITER})
        == NodeName.RENDER_FINAL_REPORT
    )
