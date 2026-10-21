"""Checkpointer serde：把 State 类型加入 msgpack 允许清单。

反序列化遇到未知自定义类型会告警，未来严格模式下直接拒绝。这里显式允许
deepresearcher 自有包内的 Pydantic 模型、Enum 与 dataclass，不对外部类开放。
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import pkgutil
from enum import Enum

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import BaseModel

# 可能进入 State 或内层循环通道的类型所在包。新增通道化类型必须落在下列包内，例如
# researcher/writer 内层通道用到的 SearchCandidate、DocumentRef、ValidatedDraft。
_PROJECT_TYPE_PACKAGES = (
    "deepresearcher.schemas",
    "deepresearcher.evidence",
    "deepresearcher.observability",
    "deepresearcher.state",
    "deepresearcher.routing",
    "deepresearcher.reporting",
    "deepresearcher.tools.web.search",
    "deepresearcher.tools.web.documents",
    "deepresearcher.agents",
)


def projected_type_keys() -> set[tuple[str, ...]]:
    keys: set[tuple[str, ...]] = set()
    for pkg_name in _PROJECT_TYPE_PACKAGES:
        try:
            pkg = importlib.import_module(pkg_name)
        except Exception:  # pragma: no cover - 包缺失不应拖垮装配
            continue
        modules = [pkg]
        path = getattr(pkg, "__path__", None)
        if path is not None:
            for info in pkgutil.walk_packages(path, pkg_name + "."):
                try:
                    modules.append(importlib.import_module(info.name))
                except Exception:  # pragma: no cover - 子模块导入失败跳过
                    continue
        for mod in modules:
            for obj in vars(mod).values():
                if not inspect.isclass(obj) or not obj.__module__.startswith("deepresearcher"):
                    continue
                is_model = (
                    issubclass(obj, BaseModel) or isinstance(obj, type) and issubclass(obj, Enum)
                )
                if not (is_model or dataclasses.is_dataclass(obj)):
                    continue
                keys.add((obj.__module__, obj.__name__))
                if getattr(obj, "__qualname__", None) and "." in obj.__qualname__:
                    keys.add((obj.__module__, obj.__qualname__))
    return keys


def build_checkpointer_serde() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=projected_type_keys())
