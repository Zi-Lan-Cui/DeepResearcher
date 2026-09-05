"""Run control-plane commands, admission, and durable queueing."""

from typing import TYPE_CHECKING, Any

from deepsearch_agent.service.runs.service import QuotaExceededError

if TYPE_CHECKING:
    from deepsearch_agent.service.runs.manager import RunManager

__all__ = ["QuotaExceededError", "RunManager"]


def __getattr__(name: str) -> Any:
    """Keep the legacy facade without loading the execution bridge for submodules."""

    if name == "RunManager":
        from deepsearch_agent.service.runs.manager import RunManager

        return RunManager
    raise AttributeError(name)
