"""LangGraph 拓扑和节点装配。"""

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from deepsearch_agent.agents import ReportWriter, ResearchAgent
from deepsearch_agent.agents.supervisor import ResearchSupervisor
from deepsearch_agent.config import Settings, get_settings
from deepsearch_agent.context import ContextPolicy
from deepsearch_agent.llm import LLMInvoker, build_llm
from deepsearch_agent.observability.instrumentation import instrument_node
from deepsearch_agent.observability.tracing.recorder import TraceRecorder
from deepsearch_agent.orchestration import nodes
from deepsearch_agent.orchestration.execution_boundary import execute_node
from deepsearch_agent.reporting import no_evidence_blockers, render_incomplete_report
from deepsearch_agent.state import ResearchState
from deepsearch_agent.tools import (
    HttpClient,
    SearchClient,
    SearchTool,
    SourceReaderTool,
    ToolConfigurationError,
    WebFetcher,
)


def _guarded_node(name, node, *, event_sink=None, trace_recorder=None, max_text_chars=1_000):
    """组合观测层与执行边界，保持两者职责独立。"""
    observed = instrument_node(
        name,
        node,
        event_sink=event_sink,
        trace_recorder=trace_recorder,
        max_text_chars=max_text_chars,
    )

    async def guarded(state):
        return await execute_node(state, stage=name, node=observed)

    return guarded


def _routed_node(
    name,
    node,
    route,
    *,
    event_sink=None,
    trace_recorder=None,
    max_text_chars=1_000,
):
    """执行节点后用 Command 动态跳转，避免条件边的隐式 fan-in 等待。"""
    guarded = _guarded_node(
        name,
        node,
        event_sink=event_sink,
        trace_recorder=trace_recorder,
        max_text_chars=max_text_chars,
    )

    async def routed(state):
        update = await guarded(state)
        target = route({**state, **update})
        return Command(update=update, goto=target)

    return routed


def _route_after_reflection(
    state: ResearchState, render_target: str = "render_final_report"
) -> str:
    """审阅通过直接渲染最终报告；只有 fatal 审阅拒绝才交回 Supervisor。"""
    run_phase = _section_value(state, "run", "phase", "routing")
    review_status = _section_value(state, "review", "status", "pending")
    if run_phase == "failed":
        return render_target
    if run_phase == "rendering":
        return render_target
    return render_target if review_status == "approved" else "supervisor"


def _route_after_node(
    state: ResearchState,
    normal_route: str,
    failure_route: str = "render_final_report",
) -> str:
    """统一把节点失败/预期终止导向最终渲染。"""
    run_phase = _section_value(state, "run", "phase", "routing")
    if run_phase == "failed":
        return failure_route
    if run_phase == "rendering" and normal_route != "writer":
        return failure_route
    return normal_route


def _section_value(state: ResearchState, name: str, field: str, default: str) -> str:
    """从状态区读取路由字段；条件边不恢复模型也不修改 State。"""
    value = state.get(name)
    if value is None:
        return default
    if isinstance(value, dict):
        return str(value.get(field, default))
    return str(getattr(value, field, default))


def _router_target(state: ResearchState) -> str:
    """把 Router 的业务路由名映射为 Graph 节点名。"""
    route = state.get("route", "deep_research")
    if route == "deep_research":
        return "clarify"
    if route == "quick_answer":
        return "quick_answer"
    return "clarify"


def build_graph(
    settings: Settings | None = None,
    *,
    llm: LLMInvoker | None = None,
    event_sink=None,
    trace_recorder: TraceRecorder | None = None,
    http_client: HttpClient | None = None,
):
    """装配完整研究应用；必需模型和联网工具缺失时立即失败。"""
    settings = settings or get_settings()
    llm = llm or build_llm(settings)
    if not settings.search.configured:
        raise ToolConfigurationError(
            "深度研究需要 BAIDU_API_KEY、TAVILY_API_KEY 或 SERPAPI_API_KEY。"
        )
    shared_http = http_client or HttpClient(settings.search)
    context_policy = ContextPolicy(
        max_tokens=settings.llm.context_window_tokens,
        summarizer=llm,
    )
    owns_http_client = http_client is None
    search_tool = SearchTool(
        SearchClient(settings.search, shared_http),
        trace_recorder=trace_recorder,
        event_sink=event_sink,
    )
    reader_tool = SourceReaderTool(
        WebFetcher(settings.search, shared_http),
        llm=llm,
        trace_recorder=trace_recorder,
        event_sink=event_sink,
        context_window_tokens=settings.llm.context_window_tokens,
        evidence_input_budget_tokens=settings.agent.evidence_input_budget_tokens,
        evidence_output_budget_tokens=settings.agent.evidence_output_budget_tokens,
        evidence_safety_margin_tokens=settings.agent.evidence_safety_margin_tokens,
        evidence_chunk_concurrency=settings.agent.evidence_chunk_concurrency,
        evidence_max_per_source=settings.agent.evidence_max_per_source,
        fetch_timeout=settings.agent.source_fetch_timeout,
        parse_timeout=settings.agent.source_parse_timeout,
        evidence_extract_timeout=settings.agent.evidence_extract_timeout,
    )
    writer_agent = ReportWriter(
        llm,
        settings.agent,
        render_incomplete=lambda state: render_incomplete_report(
            state,
            no_evidence_blockers(state),
        ),
        event_sink=event_sink,
        artifact_max_text_chars=settings.observability.max_text_chars,
        context_policy=context_policy,
    )
    research_agent = ResearchAgent(
        llm,
        settings.agent,
        search_tool=search_tool,
        reader_tool=reader_tool,
        event_sink=event_sink,
        context_policy=context_policy,
    )
    supervisor = ResearchSupervisor(
        llm,
        settings.agent,
        research_agent=research_agent,
        event_sink=event_sink,
        context_policy=context_policy,
    )
    graph = StateGraph(ResearchState)
    graph.add_node(
        "router",
        _routed_node(
            "router",
            lambda state: nodes.router(state, llm),
            lambda state: _route_after_node(state, _router_target(state), "render_final_report"),
            event_sink=event_sink,
            trace_recorder=trace_recorder,
            max_text_chars=settings.observability.max_text_chars,
        ),
        destinations=("quick_answer", "clarify", "render_final_report"),
    )
    graph.add_node(
        "clarify",
        _routed_node(
            "clarify",
            lambda state: nodes.clarify(state, llm),
            lambda state: _route_after_node(
                state,
                "render_final_report"
                if state.get("answer_mode") == "clarification_needed"
                else "supervisor",
                "render_final_report",
            ),
            event_sink=event_sink,
            trace_recorder=trace_recorder,
            max_text_chars=settings.observability.max_text_chars,
        ),
        destinations=("supervisor", "render_final_report"),
    )
    graph.add_node(
        "quick_answer",
        _routed_node(
            "quick_answer",
            lambda state: nodes.quick_answer(state, llm),
            lambda state: _route_after_node(state, "writer", "render_final_report"),
            event_sink=event_sink,
            trace_recorder=trace_recorder,
            max_text_chars=settings.observability.max_text_chars,
        ),
        destinations=("writer", "render_final_report"),
    )
    async def render_node(state):
        """渲染终点并释放由 Graph 自己创建的共享 HTTP client。"""
        try:
            return await nodes.render_final_report_node(state, llm)
        finally:
            if owns_http_client:
                await shared_http.aclose()

    graph.add_node(
        "render_final_report",
        _guarded_node(
            "render_final_report",
            render_node,
            event_sink=event_sink,
            trace_recorder=trace_recorder,
            max_text_chars=settings.observability.max_text_chars,
        ),
    )
    graph.add_node(
        "supervisor",
        _routed_node(
            "supervisor",
            supervisor.run,
            lambda state: _route_after_node(
                state, state.get("supervisor_next", "writer"), "render_final_report"
            ),
            event_sink=event_sink,
            trace_recorder=trace_recorder,
            max_text_chars=settings.observability.max_text_chars,
        ),
        destinations=("writer", "render_final_report"),
    )
    graph.add_node(
        "writer",
        _routed_node(
            "writer",
            writer_agent.run,
            lambda state: _route_after_node(
                state,
                "render_final_report"
                if _section_value(state, "writer", "status", "not_started")
                in {"failed", "exhausted"}
                or state.get("answer_mode") in {"quick_answer", "research_incomplete"}
                else "reflection",
                "render_final_report",
            ),
            event_sink=event_sink,
            trace_recorder=trace_recorder,
            max_text_chars=settings.observability.max_text_chars,
        ),
        destinations=("reflection", "render_final_report"),
    )
    graph.add_node(
        "reflection",
        _routed_node(
            "reflection",
            lambda state: nodes.reflection(state, llm),
            lambda state: _route_after_reflection(state, "render_final_report"),
            event_sink=event_sink,
            trace_recorder=trace_recorder,
            max_text_chars=settings.observability.max_text_chars,
        ),
        destinations=("supervisor", "render_final_report"),
    )
    graph.add_edge(START, "router")
    graph.add_edge("render_final_report", END)
    return graph.compile()
