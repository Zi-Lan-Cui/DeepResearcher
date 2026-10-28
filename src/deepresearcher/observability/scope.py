"""单次 Agent 调用的非持久化归属标签。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AgentScope:
    """标识这次 Agent 调用归属哪个 run/task。

    只承载坐标、不承载执行轨迹：时间、状态、耗时由中间件在调用处补进事件，
    本对象随运行时上下文注入，不进入 checkpoint。
    """

    run_id: str
    agent_name: str
    task_id: str | None = None
    parent_task_id: str | None = None
    operation_id: str | None = None

    @classmethod
    def from_task(
        cls,
        task: Mapping[str, object],
        *,
        agent_name: str,
    ) -> "AgentScope":
        task_id = str(task["id"])
        return cls(
            run_id=str(task.get("run_id") or ""),
            agent_name=agent_name,
            task_id=task_id,
            parent_task_id=str(task.get("parent_task_id") or "") or None,
            operation_id=str(task.get("operation_id") or "") or task_id,
        )

    def event_fields(self) -> dict[str, object]:
        """返回可直接合并进观测事件的稳定字段。"""
        return {
            "run_id": self.run_id,
            "agent": self.agent_name,
            "task_id": self.task_id or "",
            "parent_task_id": self.parent_task_id or "",
            "operation_id": self.operation_id or self.task_id or "",
        }
