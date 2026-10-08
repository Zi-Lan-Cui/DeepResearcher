"""内层循环 state 通道的共用 reducer。reducer 定义同一 superstep 内多份工具增量如何折叠。

同批多个 tool_calls 并发执行,并发工具只能写合并语义通道:按 id 折重、取 max、集合并、
op 流水追加、按键合并。replace 语义即整字段由最后写入值覆盖,只允许串行工具写:通道按
调用序而非完成序合并,并发写整字段快照会互相覆盖增量。各内层图共用,合并规则只在此一处。
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
    """有序 op 流水追加;活跃集按流水顺序折叠后与串行执行结果等价,并发 add 互不覆盖。"""
    return list(current or []) + list(incoming or [])


def merge_dict_by_key(current: Any, incoming: dict) -> dict:
    """按键合并;并发工具写入的 key 如候选 id、文档 id 各不相同,互不覆盖。"""
    merged = dict(current or {})
    merged.update(incoming or {})
    return merged
