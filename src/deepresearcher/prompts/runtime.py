"""向模型调用方提供每次运行的动态环境事实。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime


@dataclass(frozen=True)
class RuntimeEnvironment:
    """可注入任意 Agent/Node 的动态运行环境。"""

    current_date: str
    timezone: str

    def payload(self) -> dict[str, str]:
        return asdict(self)


def get_runtime_environment(*, now: datetime | None = None) -> RuntimeEnvironment:
    """获取结构化运行环境；不规定调用方的消息组装方式。"""
    instant = now or datetime.now().astimezone()
    return RuntimeEnvironment(
        current_date=instant.date().isoformat(),
        timezone=instant.tzname() or "local",
    )
