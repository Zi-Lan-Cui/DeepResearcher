"""内层循环 state 通道的共用 reducer。reducer 即同一 superstep 内多份工具增量的合并规则。

并发工具(同批多个 tool_calls)只能写合并语义通道:按 id 折重、取 max、集合并、op 流水
追加、按键合并;replace 语义(整字段后写覆盖)只允许串行工具写——并发快照按调用序而非
完成序合并,后写会盖掉先写。各内层图共用,同一规则只此一处。
"""

from typing import Any


def replace_channel(_current: Any, incoming: Any) -> Any:
    """整字段后写覆盖;仅串行工具写。"""
    return incoming


def retain_if_absent(current: Any, incoming: Any) -> Any:
    """incoming 为 None 时保留 current:未触及该字段的快照不清掉已提交值。"""
    return incoming if incoming is not None else current


def merge_monotonic(current: int, incoming: int) -> int:
    return max(int(current or 0), int(incoming or 0))


def merge_or(current: bool, incoming: bool) -> bool:
    return bool(current or incoming)


def append_ops(current: Any, incoming: list[dict[str, list[str]]]) -> list:
    """有序 op 流水追加;活跃集折叠后与串行执行史等价,并发 add 互不覆盖。"""
    return list(current or []) + list(incoming or [])


def merge_dict_by_key(current: Any, incoming: dict) -> dict:
    """按键合并:并发工具写入不同 key(候选/文档按 id),互不覆盖。"""
    merged = dict(current or {})
    merged.update(incoming or {})
    return merged
