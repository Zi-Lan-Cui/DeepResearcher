import asyncio

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from deepsearch_agent.agents.clarifier import Clarifier
from deepsearch_agent.agents.clarifier.graph import build_clarifier_graph
from deepsearch_agent.agents.clarifier.state import (
    ClarifierAgentState,
    ClarifierRuntimeContext,
)
from deepsearch_agent.agents.clarifier.tools import build_clarifier_tools
from deepsearch_agent.state import ResearchState


def _tool(name):
    return next(item for item in build_clarifier_tools() if item.name == name)


def test_ask_only_stages_question_without_interrupt_or_external_effect():
    runtime = type(
        "Runtime",
        (),
        {"tool_call_id": "ask-1", "state": {"query": "原问题"}},
    )()
    command = _tool("AskClarification").func(
        question="你更关心什么？",
        options=["成本", "效果", "风险"],
        runtime=runtime,
    )

    assert command.update["pending_question"] == "你更关心什么？"
    assert command.update["pending_options"] == ["成本", "效果", "风险"]
    assert command.update["clarification_rounds"] == 1
    assert not command.update.get("clarification_completed")


def test_clarifier_tools_explain_boundary_and_pause_contract():
    ask = _tool("AskClarification")
    ask_schema = ask.args_schema.model_json_schema()
    complete = _tool("ClarificationComplete")
    complete_schema = complete.args_schema.model_json_schema()

    assert "暂停当前 Run" in ask.description
    assert "研究边界不确定" in ask.description
    assert "Other" in ask_schema["properties"]["options"]["description"]
    assert "一个" in ask_schema["properties"]["question"]["description"]
    assert "唯一完成信号" in complete.description
    assert "研究边界" in complete_schema["properties"]["intent_summary"]["description"]


def test_complete_is_the_only_submission_signal():
    runtime = type("Runtime", (), {"tool_call_id": "call-2"})()

    command = _tool("ClarificationComplete").func(
        intent_summary="比较两种方案",
        research_focus=["成本", "效果"],
        assumptions=["按公开资料评估"],
        runtime=runtime,
    )

    assert command.update["clarification_completed"] is True
    assert command.update["intent_summary"] == "比较两种方案"
    assert command.update["research_focus"] == ["成本", "效果"]


def test_clarifier_run_injects_serial_tool_context():
    captured = {}

    class AgentGraph:
        async def ainvoke(self, state, *, context, config):
            captured.update(state=state, context=context, config=config)
            return {"clarification_completed": True}

    clarifier = Clarifier.__new__(Clarifier)
    clarifier.graph = AgentGraph()
    result = asyncio.run(clarifier.run({"query": "测试"}))

    assert result["clarification_completed"] is True
    assert isinstance(captured["context"], ClarifierRuntimeContext)
    assert captured["context"].tool_lock is not None
    assert captured["config"]["recursion_limit"] > 0


def test_clarifier_graph_preserves_parent_state_contract():
    agent = StateGraph(ClarifierAgentState)

    async def clarify(_state):
        return {
            "intent_summary": "已确认的研究范围",
            "clarification_completed": True,
        }

    agent.add_node("complete", clarify)
    agent.add_edge(START, "complete")
    agent.add_edge("complete", END)
    result = asyncio.run(
        build_clarifier_graph(agent.compile().ainvoke).ainvoke({"query": "原问题"})
    )

    assert result["query"] == "原问题"
    assert result["clarified_query"] == "原问题"
    assert result["research_brief"] == "已确认的研究范围"


def test_clarifier_subgraph_interrupt_resumes_through_parent_checkpoint():
    async def scripted_agent(state):
        answered = any(
            isinstance(message, HumanMessage) and "【用户澄清回答】" in str(message.content)
            for message in state.get("messages", [])
        )
        if answered:
            return {
                "intent_summary": "评估方案效果",
                "research_focus": ["效果"],
                "clarification_completed": True,
            }
        return {
            "pending_question": "你更关心哪一方面？",
            "pending_options": ["成本", "效果", "风险"],
            "clarification_rounds": 1,
        }

    child = build_clarifier_graph(scripted_agent)
    parent = StateGraph(ResearchState)
    parent.add_node("clarifier", child)
    parent.add_edge(START, "clarifier")
    parent.add_edge("clarifier", END)
    compiled = parent.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "clarifier-hitl"}}

    first = asyncio.run(compiled.ainvoke({"query": "比较两个方案"}, config))
    interrupts = first.get("__interrupt__", ())
    assert interrupts
    assert interrupts[0].value["options"] == ["成本", "效果", "风险"]

    resumed = asyncio.run(compiled.ainvoke(Command(resume={"answer": "效果"}), config))
    assert resumed["clarified_query"] == "比较两个方案"
    assert "评估方案效果" in resumed["research_brief"]
