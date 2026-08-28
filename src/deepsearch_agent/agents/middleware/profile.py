"""中间件栈的声明式配置。

收拢 ``build_agent_middleware`` 的长参数列表：每个 Agent 用一份 Profile
描述自己要的中间件栈，参数间的成对约束（如提交守卫的消息与探测）
由结构本身保证，而不是靠两个可空参数之间的隐式约定。
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel


@dataclass(frozen=True)
class SubmissionGuard:
    """工具提交守卫配置：成对出现，杜绝只配一半。

    模型在完成提交（``submitted_probe`` 返回 False）前输出纯文本时，
    注入 ``nudge_message`` 并踢回重试；耗尽后放行给业务层兜底。
    """

    nudge_message: str
    submitted_probe: Callable[[Any], bool]


@dataclass(frozen=True)
class MiddlewareProfile:
    """一个 Agent 的中间件栈完整输入。

    ``max_turns`` 只是防失控的模型调用天花板，不承担业务配额语义；
    业务配额由 ``tool_call_limits``（单工具调用数）与业务层 hard check 表达。
    """

    agent_name: str
    max_turns: int
    context_window_tokens: int
    model: BaseChatModel | None = None
    retry_tools: Sequence[tuple[list[str], str]] = ()
    serial_tools: set[str] | None = None
    tool_call_limits: Sequence[tuple[str, int]] = ()
    submission_guard: SubmissionGuard | None = None
    emit: Callable[[str, dict[str, object]], None] | None = None
