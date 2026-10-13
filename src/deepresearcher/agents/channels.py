"""内层循环 state 通道的共用 reducer。reducer 定义同一 superstep 内多份工具增量如何折叠。

同批多个 tool_calls 并发执行,并发工具只能写合并语义通道:按 id 折重、取 max、集合并、
差量施加到集合、按键合并。replace 语义即整字段由最后写入值覆盖,只允许串行工具写:通道按
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


def merge_active_set(current: Any, incoming: dict[str, list[str]]) -> list[str]:
    """活跃工作集通道的合并:通道值是集合本身(排序列表),incoming 是 {add, remove} 差量。

    并发工具各交自己的变化,合并结果与施加顺序无关(同批只有 add,remove 来自独占批);
    入口种子同样按差量写入(对空集 add 全量)。读取方直接用通道现值,不重放历史。
    """
    values = set(current or [])
    values.update(incoming.get("add", []))
    values.difference_update(incoming.get("remove", []))
    return sorted(values)


def merge_dict_by_key(current: Any, incoming: dict) -> dict:
    """按键合并;并发工具写入的 key 如候选 id、文档 id 各不相同,互不覆盖。"""
    merged = dict(current or {})
    merged.update(incoming or {})
    return merged


def merge_registry_by_key(current: Any, incoming: dict[str, Any]) -> dict:
    """注册表通道的合并:通道值是登记表本身,incoming 是 {add, remove} 差量。

    同批并发派发各自 add 不同的键互不覆盖;remove 只出现在独占批(吸收或丢失判定),
    与并发 add 不交叠。读取方直接用通道现值。
    """
    merged = dict(current or {})
    merged.update((incoming or {}).get("add") or {})
    for key in (incoming or {}).get("remove") or []:
        merged.pop(key, None)
    return merged
