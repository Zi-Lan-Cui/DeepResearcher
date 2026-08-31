"""LangGraph 拓扑和节点装配。"""

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from deepsearch_agent.agents import Clarifier, ReportWriter, ResearchAgent
from deepsearch_agent.agents.clarifier.graph import build_clarifier_graph
from deepsearch_agent.agents.supervisor import ResearchSupervisor
from deepsearch_agent.agents.writer.graph import build_writer_graph
from deepsearch_agent.config import Settings, get_settings
from deepsearch_agent.llm import LLMInvoker, build_llm
from deepsearch_agent.observability.instrumentation import instrument_node
from deepsearch_agent.observability.tracing.recorder import TraceRecorder
from deepsearch_agent.orchestration import nodes
from deepsearch_agent.orchestration.execution_boundary import execute_node
from deepsearch_agent.reporting import no_evidence_blockers, render_incomplete_report
from deepsearch_agent.routing import (
    NodeName,
    route_after_clarify,
    route_after_quick_answer,
    route_after_reflection,
    route_after_router,
    route_after_supervisor,
    route_after_writer,
)
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


def build_graph(
    settings: Settings | None = None,
    *,
    llm: LLMInvoker | None = None,
    event_sink=None,
    trace_recorder: TraceRecorder | None = None,
    http_client: HttpClient | None = None,
    checkpointer=None,
):
    """装配完整研究应用；必需模型和联网工具缺失时立即失败。

    checkpointer 为 LangGraph BaseCheckpointSaver（如 AsyncPostgresSaver）：
    每个 superstep 结束持久化 state 通道，调用方以
    config={"configurable": {"thread_id": run_id}} 获得断点重放/续跑能力；
    None（CLI/测试默认）行为与既往完全一致。
    """
    settings = settings or get_settings()
    llm = llm or build_llm(settings)

    if not settings.search.configured:
        raise ToolConfigurationError(
            "深度研究需要 BAIDU_API_KEY、TAVILY_API_KEY 或 SERPAPI_API_KEY。"
        )

    shared_http = http_client or HttpClient(settings.search)
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
    clarifier = Clarifier(
        llm,
        settings.agent,
        context_window_tokens=settings.llm.context_window_tokens,
    )
    clarifier_graph = build_clarifier_graph(clarifier.run)
    writer_agent = ReportWriter(
        llm,
        settings.agent,
        render_incomplete=lambda state: render_incomplete_report(
            state,
            no_evidence_blockers(state),
        ),
        event_sink=event_sink,
        artifact_max_text_chars=settings.observability.max_text_chars,
        context_window_tokens=settings.llm.context_window_tokens,
    )
    writer_graph = build_writer_graph(writer_agent.run)
    research_agent = ResearchAgent(
        llm,
        settings.agent,
        search_tool=search_tool,
        reader_tool=reader_tool,
        event_sink=event_sink,
        context_window_tokens=settings.llm.context_window_tokens,
    )
    supervisor = ResearchSupervisor(
        llm,
        settings.agent,
        research_agent=research_agent,
        event_sink=event_sink,
        context_window_tokens=settings.llm.context_window_tokens,
    )
    graph = StateGraph(ResearchState)
    graph.add_node(
        NodeName.ROUTER,
        _routed_node(
            NodeName.ROUTER,
            lambda state: nodes.router(state, llm),
            route_after_router,
            event_sink=event_sink,
            trace_recorder=trace_recorder,
            max_text_chars=settings.observability.max_text_chars,
        ),
        destinations=(
            NodeName.QUICK_ANSWER,
            NodeName.CLARIFY,
            NodeName.RENDER_FINAL_REPORT,
        ),
    )
    graph.add_node(
        NodeName.CLARIFY,
        _routed_node(
            NodeName.CLARIFY,
            clarifier_graph.ainvoke,
            route_after_clarify,
            event_sink=event_sink,
            trace_recorder=trace_recorder,
            max_text_chars=settings.observability.max_text_chars,
        ),
        destinations=(NodeName.SUPERVISOR,),
    )
    graph.add_node(
        NodeName.QUICK_ANSWER,
        _routed_node(
            NodeName.QUICK_ANSWER,
            lambda state: nodes.quick_answer(state, llm),
            route_after_quick_answer,
            event_sink=event_sink,
            trace_recorder=trace_recorder,
            max_text_chars=settings.observability.max_text_chars,
        ),
        destinations=(NodeName.WRITER, NodeName.RENDER_FINAL_REPORT),
    )

    async def render_node(state):
        """渲染终点并释放由 Graph 自己创建的共享 HTTP client。"""
        try:
            return await nodes.render_final_report_node(state)
        finally:
            if owns_http_client:
                await shared_http.aclose()

    graph.add_node(
        NodeName.RENDER_FINAL_REPORT,
        _guarded_node(
            NodeName.RENDER_FINAL_REPORT,
            render_node,
            event_sink=event_sink,
            trace_recorder=trace_recorder,
            max_text_chars=settings.observability.max_text_chars,
        ),
    )
    graph.add_node(
        NodeName.SUPERVISOR,
        _routed_node(
            NodeName.SUPERVISOR,
            supervisor.run,
            route_after_supervisor,
            event_sink=event_sink,
            trace_recorder=trace_recorder,
            max_text_chars=settings.observability.max_text_chars,
        ),
        destinations=(NodeName.WRITER, NodeName.RENDER_FINAL_REPORT),
    )
    graph.add_node(
        NodeName.WRITER,
        _routed_node(
            NodeName.WRITER,
            writer_graph.ainvoke,
            route_after_writer,
            event_sink=event_sink,
            trace_recorder=trace_recorder,
            max_text_chars=settings.observability.max_text_chars,
        ),
        destinations=(NodeName.REFLECTION, NodeName.RENDER_FINAL_REPORT),
    )
    graph.add_node(
        NodeName.REFLECTION,
        _routed_node(
            NodeName.REFLECTION,
            lambda state: nodes.reflection(state, llm),
            route_after_reflection,
            event_sink=event_sink,
            trace_recorder=trace_recorder,
            max_text_chars=settings.observability.max_text_chars,
        ),
        destinations=(NodeName.SUPERVISOR, NodeName.RENDER_FINAL_REPORT),
    )
    graph.add_edge(START, NodeName.ROUTER)
    graph.add_edge(NodeName.RENDER_FINAL_REPORT, END)
    return graph.compile(checkpointer=checkpointer)
