"""Writer 子图装配。"""

from collections.abc import Awaitable, Callable

from langgraph.graph import END, START, StateGraph

from deepresearcher.state import ResearchState

WriterNode = Callable[[ResearchState], Awaitable[dict[str, object]]]


def build_writer_graph(writer: WriterNode):
    """构建 Writer 子图，保持顶层状态契约不变。"""
    graph = StateGraph(ResearchState)

    async def write(state: ResearchState) -> dict[str, object]:
        return await writer(state)

    graph.add_node("write", write)
    graph.add_edge(START, "write")
    graph.add_edge("write", END)
    return graph.compile()
