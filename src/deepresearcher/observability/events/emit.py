"""Agent/领域审计事件的一行式发射。

日志与 sink 落盘的顺序在此收敛为一份实现；上下文关联字段由事件工厂
自读 contextvars，各 Agent 的调用方无需自行展开 ``current_context()``。
"""

import logging
from collections.abc import Mapping
from typing import Protocol

from deepresearcher.observability.events.models import make_audit_event
from deepresearcher.observability.events.sink import JsonlSink

# 默认视为"完整正文"的 payload 键：只保留长度与受限预览进入事件流。
DEFAULT_CONTENT_KEYS = ("markdown", "normalized_markdown", "report")


class AgentEmit(Protocol):
    """各 Agent Deps 携带的事件发射回调的统一契约。

    绑定自 agents 侧的 ``_emit``/``_emit_audit_event``。``component`` 关键字允许
    领域事件改记到别的组件名下，例如 Supervisor 派发链发出的事件归属
    research_agent 组件，默认值由绑定方自带。Deps 以此协议替代宽松的
    Callable，两个 agent 的 emit 遵循同一契约。
    """

    def __call__(
        self,
        event_type: str,
        payload: dict[str, object],
        *,
        component: str = ...,
    ) -> None: ...


def bounded_content(
    payload: Mapping[str, object],
    *,
    max_text_chars: int,
    content_keys: tuple[str, ...] = DEFAULT_CONTENT_KEYS,
) -> dict[str, object]:
    """把长正文键替换为 ``*_chars`` 与 ``*_preview``，防止整篇产物进入事件流。"""
    event_payload = {key: value for key, value in payload.items() if key not in content_keys}
    for key in content_keys:
        if key in payload:
            value = str(payload[key])
            event_payload[f"{key}_chars"] = len(value)
            event_payload[f"{key}_preview"] = value[:max_text_chars]
    return event_payload


def emit_agent_event(
    event_sink: JsonlSink | None,
    logger: logging.Logger,
    event_type: str,
    payload: dict[str, object],
    *,
    component: str,
    node_fallback: str | None = None,
) -> None:
    """记录 Agent 生命周期事件：先入日志，再按需写 JSONL sink。

    可观测性是旁路：日志 handler 或 sink 落盘的异常一律在此吞掉并转成错误日志,
    不阻断、重试或改写宿主执行。保护统一做在这一层,各调用点不必再各自做防护。
    """
    try:
        logger.info("%s payload=%s", event_type, payload)
        if event_sink is not None:
            event_sink.write(
                make_audit_event(
                    event_type,
                    component=component,
                    node_id_fallback=node_fallback,
                    payload=payload,
                )
            )
    except Exception:
        logger.exception("%s emission failed", event_type)
