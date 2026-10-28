"""单个研究任务：Supervisor 派发、Researcher 接收、工具读取三方共用。"""

from typing import NotRequired, TypedDict


class Task(TypedDict):
    """Supervisor 派给一个 Researcher 的研究任务；一次 run 下有多个 task。"""

    id: str
    run_id: NotRequired[str]
    question: str
    # 任务卡展示短题，由 Supervisor 派发时给出；question 是完整研究契约，两者分离。
    display_title: NotRequired[str]
    round: NotRequired[int]
    sequence: NotRequired[int]
    worker_id: NotRequired[str]
    worker_index: NotRequired[int]
    parent_task_id: NotRequired[str]
    operation_id: NotRequired[str]
