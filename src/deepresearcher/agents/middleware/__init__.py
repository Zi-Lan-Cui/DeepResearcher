"""Agent 共享中间件与栈装配。"""

from deepresearcher.agents.middleware.factory import (
    AGENT_RECURSION_LIMIT,
    MiddlewareProfile,
    SubmissionGuard,
    build_agent_middleware,
)
from deepresearcher.agents.middleware.observability import (
    LIMIT_MESSAGE_MARKER,
    AgentObservabilityMiddleware,
)
from deepresearcher.agents.middleware.retry import model_retry, tool_retry
from deepresearcher.agents.middleware.serial_tools import SerialToolMiddleware
from deepresearcher.agents.middleware.state_rebuild import LoopStateRebuildMiddleware
from deepresearcher.agents.middleware.tool_loop_guard import ToolLoopGuardMiddleware

__all__ = [
    "AGENT_RECURSION_LIMIT",
    "AgentObservabilityMiddleware",
    "LIMIT_MESSAGE_MARKER",
    "LoopStateRebuildMiddleware",
    "MiddlewareProfile",
    "SerialToolMiddleware",
    "SubmissionGuard",
    "ToolLoopGuardMiddleware",
    "build_agent_middleware",
    "model_retry",
    "tool_retry",
]
