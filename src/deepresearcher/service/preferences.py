"""个人研究偏好：选项枚举、提交校验、创建冻结与对 AgentConfig 的应用。"""

from dataclasses import replace
from typing import Any

from deepresearcher.config import AgentConfig

PREFERENCE_CHOICES: dict[str, list[int]] = {
    "max_research_rounds": [1, 2, 3, 4],
    "max_tasks_per_round": [4, 8, 12],
    "research_agent_max_turns": [8, 12, 20],
    "supervisor_max_active_evidences": [20, 30, 50],
}


def clamp_preferences(raw: dict[str, Any]) -> dict[str, int]:
    """校验整份提交：必须恰好覆盖全部键、值落在枚举内，否则 ValueError（路由层转 422）。"""
    if set(raw) != set(PREFERENCE_CHOICES):
        raise ValueError(f"偏好必须整份提交，键需恰为 {sorted(PREFERENCE_CHOICES)}")
    clamped: dict[str, int] = {}
    for field_name, value in raw.items():
        choices = PREFERENCE_CHOICES[field_name]
        if isinstance(value, bool) or not isinstance(value, int) or value not in choices:
            raise ValueError(f"{field_name} 取值非法：{value}；可选 {choices}")
        clamped[field_name] = value
    return clamped


def frozen_snapshot(user_preferences: dict[str, Any] | None, agent: AgentConfig) -> dict[str, int]:
    """创建时刻的全量快照：用户显式项优先，未落项与越界脏值都回服务器默认。"""
    snapshot: dict[str, int] = {}
    for field_name, choices in PREFERENCE_CHOICES.items():
        server_default = getattr(agent, field_name)
        value = (user_preferences or {}).get(field_name, server_default)
        snapshot[field_name] = value if value in choices else server_default
    return snapshot


def apply_preferences(agent: AgentConfig, snapshot: dict[str, Any] | None) -> AgentConfig:
    """把 run 快照应用为 per-run AgentConfig；逐项复核枚举，越界项静默落回服务器值。"""
    overrides = {
        field_name: value
        for field_name, choices in PREFERENCE_CHOICES.items()
        if snapshot
        and (value := snapshot.get(field_name)) is not None
        and not isinstance(value, bool)
        and value in choices
    }
    return replace(agent, **overrides) if overrides else agent
