"""跨 agent 共用的中间件与栈装配。

只收被两个以上 agent 使用的中间件；绑定单一 agent 上下文槽的中间件（例如
Supervisor 的任务结果接收）住在该 agent 包的 middleware.py，避免本包反向依赖业务层。
"""

from deepresearcher.agents.middleware.agent_events import (
    LIMIT_MESSAGE_MARKER,
    AgentEventsMiddleware,
)
from deepresearcher.agents.middleware.factory import (
    AGENT_RECURSION_LIMIT,
    MiddlewareProfile,
    SubmissionGuard,
    build_agent_middleware,
)
from deepresearcher.agents.middleware.retry import model_retry, tool_retry
from deepresearcher.agents.middleware.serial_tools import SerialToolMiddleware
from deepresearcher.agents.middleware.state_rebuild import StateRebuildMiddleware
from deepresearcher.agents.middleware.tool_loop_guard import ToolLoopGuardMiddleware

__all__ = [
    "AGENT_RECURSION_LIMIT",
    "AgentEventsMiddleware",
    "LIMIT_MESSAGE_MARKER",
    "MiddlewareProfile",
    "SerialToolMiddleware",
    "StateRebuildMiddleware",
    "SubmissionGuard",
    "ToolLoopGuardMiddleware",
    "build_agent_middleware",
    "model_retry",
    "tool_retry",
]
