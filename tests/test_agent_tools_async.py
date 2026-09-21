import inspect

from deepresearcher.agents.clarifier.tools import build_clarifier_tools
from deepresearcher.agents.researcher.tools import build_researcher_tools
from deepresearcher.agents.supervisor.tools import build_supervisor_tools
from deepresearcher.agents.writer.tools import build_writer_tools


def test_all_agent_tools_use_native_async_entrypoints():
    tools = [
        *build_clarifier_tools(),
        *build_researcher_tools(),
        *build_supervisor_tools(),
        *build_writer_tools(turn_budget=8, read_batch=30),
    ]

    assert tools
    for tool in tools:
        assert tool.coroutine is not None, tool.name
        assert inspect.iscoroutinefunction(tool.coroutine), tool.name
