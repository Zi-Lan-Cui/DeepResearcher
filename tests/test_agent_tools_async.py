import inspect

from deepsearch_agent.agents.clarifier.tools import build_clarifier_tools
from deepsearch_agent.agents.researcher.tools import build_researcher_tools
from deepsearch_agent.agents.supervisor.tools import build_supervisor_tools
from deepsearch_agent.agents.writer.tools import build_writer_tools


def test_all_agent_tools_use_native_async_entrypoints():
    tools = [
        *build_clarifier_tools(),
        *build_researcher_tools(),
        *build_supervisor_tools(),
        *build_writer_tools(),
    ]

    assert tools
    for tool in tools:
        assert tool.coroutine is not None, tool.name
        assert inspect.iscoroutinefunction(tool.coroutine), tool.name
