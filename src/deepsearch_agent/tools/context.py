"""工具调用的稳定身份上下文。

这里只定义归属与未来幂等接口，不在本层实现缓存或去重策略。
"""

from __future__ import annotations

from dataclasses import dataclass

from deepsearch_agent.state import SubTask


@dataclass(frozen=True, slots=True)
class ToolExecutionContext:
    """一次方向内工具执行的稳定身份。

    ``task_id`` 表示研究方向归属；``operation_id`` 预留给逻辑调用去重。
    二者不能混用：同一 task 可以包含多次不同的工具操作。
    """

    run_id: str
    task_id: str
    operation_id: str | None = None
    parent_task_id: str | None = None

    @classmethod
    def from_task(cls, task: SubTask) -> "ToolExecutionContext":
        task_id = task["id"]
        return cls(
            run_id=str(task.get("run_id") or ""),
            task_id=task_id,
            operation_id=str(task.get("operation_id") or "") or None,
            parent_task_id=str(task.get("parent_task_id") or "") or None,
        )

    def event_fields(self) -> dict[str, str]:
        """返回可进入内部事件的身份字段。"""
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "operation_id": self.operation_id or self.task_id,
            "parent_task_id": self.parent_task_id or "",
        }
