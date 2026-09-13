"""LangChain 工具调用历史的协议检查与安全分组。"""

from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage


class ToolMessageProtocolError(ValueError):
    """消息历史中的工具调用没有得到正确回补。"""


def validate_tool_call_pairs(messages: Sequence[BaseMessage]) -> None:
    """确认每个带 tool_calls 的 AIMessage 都有匹配的 ToolMessage。"""
    pending: set[str] = set()
    for message in messages:
        if isinstance(message, AIMessage):
            if pending:
                raise ToolMessageProtocolError(
                    f"前一条 AIMessage 的工具调用未完成：{sorted(pending)}"
                )
            pending = {
                str(call.get("id", "")) for call in (message.tool_calls or []) if call.get("id")
            }
        elif isinstance(message, ToolMessage):
            call_id = str(message.tool_call_id)
            if call_id not in pending:
                raise ToolMessageProtocolError(f"ToolMessage 没有对应的 tool_call_id：{call_id}")
            pending.remove(call_id)
    if pending:
        raise ToolMessageProtocolError(f"存在未回补的工具调用：{sorted(pending)}")


def message_groups(messages: Sequence[BaseMessage]) -> list[list[BaseMessage]]:
    """按不可拆分的工具事务分组，供上下文裁剪使用。"""
    groups: list[list[BaseMessage]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if isinstance(message, AIMessage) and message.tool_calls:
            expected = {str(call.get("id", "")) for call in message.tool_calls if call.get("id")}
            group: list[BaseMessage] = [message]
            index += 1
            while index < len(messages) and expected:
                current = messages[index]
                group.append(current)
                if isinstance(current, ToolMessage):
                    expected.discard(str(current.tool_call_id))
                index += 1
            groups.append(group)
            continue
        groups.append([message])
        index += 1
    return groups
