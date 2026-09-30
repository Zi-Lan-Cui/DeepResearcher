"""LangGraph 拓扑和节点装配。"""

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from deepresearcher import nodes
from deepresearcher.agents import Clarifier, ReportWriter, ResearchAgent
from deepresearcher.agents.clarifier.graph import build_clarifier_graph
from deepresearcher.agents.supervisor import ResearchSupervisor
from deepresearcher.config import Settings, get_settings
from deepresearcher.errors import clip_text
from deepresearcher.llm import build_llm
from deepresearcher.node_runner import execute_node
from deepresearcher.observability.instrumentation import instrument_node
from deepresearcher.reporting import no_evidence_blockers, render_incomplete_report
from deepresearcher.routing import (
    NodeName,
    route_after_clarify,
    route_after_quick_answer,
    route_after_reviewer,
    route_after_router,
    route_after_supervisor,
    route_after_writer,
)
from deepresearcher.state import ResearchState
from deepresearcher.tools import (
    AliyunFetchProvider,
    DirectHttpFetchProvider,
    FetchService,
    HttpClient,
    SearchService,
    SearchTool,
    SourceReaderTool,
    ToolConfigurationError,
)
from deepresearcher.tools.transport.aliyun import create_aliyun_dts_client
from deepresearcher.tools.web.materials import MemoryResearchMaterialStore


def summarize_node_result(result: dict, *, max_text_chars: int) -> dict:
    """记录运行元数据和有界预览；完整正文不进入事件流或普通日志。

    这里认识的是 State 通道词表(装配层职责)，观测层 instrumentation 不解读业务字段;
    projector 的面向用户帧从这里登记的键读取。
    """
    summary: dict = {"updated_fields": sorted(result.keys())}
    for key in (
        "route",
        "answer_mode",
        "evidence_count",
        "source_count",
    ):
        if key in result:
            summary[key] = result[key]
    if "route_reason" in result:
        summary["route_reason"] = str(result["route_reason"])[:240]
    if "clarified_query" in result:
        summary["clarified_query"] = str(result["clarified_query"])[:300]
    if "research_brief" in result:
        summary["research_brief"] = str(result["research_brief"])[:max_text_chars]
    run = result.get("run")
    if run is not None:
        summary["phase"] = getattr(run, "phase", None)
        summary["terminal_reason"] = getattr(run, "terminal_reason", "")
    for key in ("supervisor", "writer", "review"):
        section_value = result.get(key)
        if section_value is not None:
            summary[f"{key}_status"] = getattr(section_value, "status", None)
            if key == "supervisor":
                summary["current_round"] = getattr(section_value, "current_round", 0)
            if key in {"writer", "review"}:
                feedback = getattr(section_value, "feedback", "")
                if feedback:
                    summary[f"{key}_feedback"] = clip_text(str(feedback))
    if "rejected_draft" in result:
        text = str(result["rejected_draft"])
        summary["rejected_draft_chars"] = len(text)
        summary["rejected_draft_preview"] = text[:max_text_chars]
    review = result.get("review")
    review_issues = getattr(review, "issues", None) if review is not None else None
    if review_issues:
        summary["review_issues"] = [
            {
                "severity": item.severity,
                "claim": item.claim[:200],
                "reason": item.reason[:300],
            }
            for item in review_issues
        ]
    if "report" in result:
        text = str(result["report"])
        summary["report_chars"] = len(text)
        summary["report_preview"] = text[:max_text_chars]
    directive_brief = getattr(result.get("writer_directive"), "report_brief", None)
    if directive_brief is not None:
        summary["report_brief"] = str(directive_brief)[:max_text_chars]
    research = result.get("supervisor")
    coverage_gaps = getattr(research, "coverage_gaps", None) if research is not None else None
    if coverage_gaps:
        summary["coverage_gaps"] = list(coverage_gaps)
    for key in (
        "citations",
        "task_results",
        "evidences",
        "paragraph_bindings",
    ):
        if key in result:
            summary[f"{key}_count"] = len(result[key])
    return summary


def _guarded_node(name, node, *, event_sink=None, max_text_chars: int):
    """组合观测层与节点运行器（node_runner），保持两者职责独立。"""
    observed = instrument_node(
        name,
        node,
        event_sink=event_sink,
        max_text_chars=max_text_chars,
        summarize=lambda value: summarize_node_result(value, max_text_chars=max_text_chars),
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
    max_text_chars: int,
):
    """执行节点后用 Command 动态跳转，避免条件边的隐式 fan-in 等待。"""
    guarded = _guarded_node(
        name,
        node,
        event_sink=event_sink,
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
    llm: BaseChatModel | None = None,
    event_sink=None,
    http_client: HttpClient | None = None,
    checkpointer=None,
    material_store=None,
    provider_health=None,
):
    """装配完整研究应用；必需模型和联网工具缺失时立即失败。

    checkpointer 为 LangGraph BaseCheckpointSaver（如 AsyncPostgresSaver）：
    每个 superstep 结束持久化 state 通道，调用方以
    config={"configurable": {"thread_id": run_id}} 获得断点重放/续跑能力；
    None（测试/直接库调用默认）行为与既往完全一致。
    """
    settings = settings or get_settings()
    llm = llm or build_llm(settings)

    if not settings.search.configured:
        provider = settings.search.provider
        if provider == "auto":
            detail = (
                "需要 BAIDU/TAVILY/SERPAPI API Key；如使用阿里云，请显式设置 "
                "SEARCH_PROVIDER=aliyun 并配置默认凭据链"
            )
        else:
            detail = f"已选择 {provider}，但未配置该搜索 Provider 所需的凭据"
        raise ToolConfigurationError(f"深度研究搜索不可用：{detail}。")

    shared_http = http_client or HttpClient(settings.search)
    shared_materials = material_store or MemoryResearchMaterialStore()
    owns_http_client = http_client is None
    uses_aliyun = settings.search.provider == "aliyun" or (
        "aliyun" in settings.search.fetch_provider_order
    )
    aliyun_client = create_aliyun_dts_client(settings.search) if uses_aliyun else None
    search_tool = SearchTool(
        SearchService(
            settings.search,
            shared_http,
            aliyun_client=aliyun_client,
            provider_health=provider_health,
        ),
        event_sink=event_sink,
    )
    fetch_providers = []
    for provider_name in settings.search.fetch_provider_order:
        if provider_name == "aliyun":
            if aliyun_client is None:  # pragma: no cover - guarded by uses_aliyun
                raise ToolConfigurationError("未初始化阿里云 DTS AI 客户端")
            fetch_providers.append(AliyunFetchProvider(settings.search, aliyun_client))
        else:
            fetch_providers.append(DirectHttpFetchProvider(settings.search, shared_http))
    reader_tool = SourceReaderTool(
        FetchService(
            fetch_providers,
        ),
        event_sink=event_sink,
        fetch_timeout=settings.agent.source_fetch_timeout,
        parse_timeout=settings.agent.source_parse_timeout,
        material_store=shared_materials,
        document_inline_max_tokens=settings.agent.document_inline_max_tokens,
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
    research_agent = ResearchAgent(
        llm,
        settings.agent,
        search_tool=search_tool,
        reader_tool=reader_tool,
        event_sink=event_sink,
        context_window_tokens=settings.llm.context_window_tokens,
        material_store=shared_materials,
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
            lambda state: nodes.router(state, llm, agent_config=settings.agent),
            route_after_router,
            event_sink=event_sink,
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
            max_text_chars=settings.observability.max_text_chars,
        ),
        # clarification_needed 与 terminal phase 都直落渲染:destinations 是图形状
        # 唯一机器可读声明,漏 RENDER 会误导读者与绘图工具。
        destinations=(NodeName.SUPERVISOR, NodeName.RENDER_FINAL_REPORT),
    )
    graph.add_node(
        NodeName.QUICK_ANSWER,
        _routed_node(
            NodeName.QUICK_ANSWER,
            lambda state: nodes.quick_answer(state, llm, agent_config=settings.agent),
            route_after_quick_answer,
            event_sink=event_sink,
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
            max_text_chars=settings.observability.max_text_chars,
        ),
        destinations=(NodeName.WRITER, NodeName.RENDER_FINAL_REPORT),
    )
    graph.add_node(
        NodeName.WRITER,
        _routed_node(
            NodeName.WRITER,
            writer_agent.run,
            route_after_writer,
            event_sink=event_sink,
            max_text_chars=settings.observability.max_text_chars,
        ),
        destinations=(NodeName.REVIEWER, NodeName.RENDER_FINAL_REPORT),
    )
    graph.add_node(
        NodeName.REVIEWER,
        _routed_node(
            NodeName.REVIEWER,
            lambda state: nodes.reviewer(state, llm, agent_config=settings.agent),
            route_after_reviewer,
            event_sink=event_sink,
            max_text_chars=settings.observability.max_text_chars,
        ),
        destinations=(NodeName.SUPERVISOR, NodeName.RENDER_FINAL_REPORT),
    )
    graph.add_edge(START, NodeName.ROUTER)
    graph.add_edge(NodeName.RENDER_FINAL_REPORT, END)
    return graph.compile(checkpointer=checkpointer)
