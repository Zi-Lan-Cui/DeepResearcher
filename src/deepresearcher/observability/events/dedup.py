"""事件去重键：只由业务对象与执行位置构成，重放时取到同一个值。

{checkpoint_ns}|{event_type}|{identity}   位置与对象都可得
{event_type}|{identity}                   仅有对象身份
都不成立 → None                           该事件不参与去重
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from langchain_core.runnables import ensure_config

#: 组成对象身份的 payload 键；取全部非空值拼接，顺序固定。
#:
#: task_id 与 operation_id 在一次任务内恒定，只有这两项时，同一任务内同类事件的多次
#: 发生会算出同一个键，写库时被判为重放而丢弃。对象标识键补齐区分度：检索用 query，
#: 抓取用 requested_url 或 url，文档登记用 document_id。这些值取自模型输出、搜索
#: 结果或正文内容，重放同一次执行时取值不变，不影响去重语义。
_IDENTITY_KEYS = (
    "task_id",
    "tool_call_id",
    "operation_id",
    "document_id",
    "url",
    "requested_url",
    "query",
)

#: 键长上限，与 run_events.dedup_key 列宽（512）留有一倍余量。超长改用摘要，
#: 避免写库时被列宽截断，把两条不同的键截成同一个。
_DEDUP_KEY_MAX_CHARS = 256


@dataclass(frozen=True)
class ExecutionPosition:
    """当前事件所属的执行位置：事件发出处所在的图空间。"""

    namespace: str


_bound_position: ContextVar[ExecutionPosition | None] = ContextVar(
    "deepresearcher_bound_position", default=None
)


@contextmanager
def bind_position(namespace: str | None) -> Iterator[None]:
    """在作用域内绑定执行位置；空值时沿用外层，不覆盖。"""
    if not namespace:
        yield
        return
    token = _bound_position.set(ExecutionPosition(namespace))
    try:
        yield
    finally:
        _bound_position.reset(token)


@contextmanager
def bind_position_from(config: Any) -> Iterator[None]:
    """从 LangGraph 注入的 config 取出 checkpoint_ns 并绑定。"""
    configurable = config.get("configurable") if isinstance(config, Mapping) else None
    if not isinstance(configurable, Mapping):
        configurable = {}
    with bind_position(_string_or_empty(configurable.get("checkpoint_ns"))):
        yield


def bind_position_from_current() -> Any:
    """按当前上下文里的 config 绑定位置，返回 contextmanager。"""
    try:
        config = ensure_config()
    except Exception:  # noqa: BLE001 - 取不到就是没位置，降级处理，不阻断执行
        config = None
    return bind_position_from(config)


def _string_or_empty(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _object_identity(payload: Mapping[str, Any] | None) -> str:
    """取业务对象身份；多个键拼接，同一 task 下的不同工具调用因此不互相覆盖。"""
    if not payload:
        return ""
    found = [_string_or_empty(payload.get(key)) for key in _IDENTITY_KEYS]
    return "/".join(value for value in found if value)


def build_dedup_key(
    event_type: str,
    payload: Mapping[str, Any] | None = None,
) -> str | None:
    """构造重放不变的去重键；信息不足以定身份时返回 None。"""
    identity = _object_identity(payload)
    position = _bound_position.get()
    if position is not None:
        parts = (position.namespace, event_type, identity)
    elif identity:
        parts = (event_type, identity)
    else:
        return None
    key = "|".join(parts)
    if len(key) <= _DEDUP_KEY_MAX_CHARS:
        return key
    return f"h:{sha256(key.encode('utf-8')).hexdigest()}"
