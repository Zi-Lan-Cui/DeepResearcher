"""Checkpointer serde：把**我们自己的** State 类型加入 msgpack 允许清单。

LangGraph 把每个 superstep 的 state 以 msgpack 存进 Postgres；反序列化时对未知自定义
类型默认告警、未来严格模式会**直接拒绝**——这会让"崩溃恢复 / 澄清 resume"在升级后炸。
这里显式放行 deepresearcher 自有包里的 Pydantic 模型 / Enum / dataclass（即可能进
ResearchState 的通道类型），既消除告警、又不对任意（外部）类开放反序列化。

扫描式收集：以后新增 state 类型只要落在这些包内即自动覆盖，无需手维护清单。
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import pkgutil
from enum import Enum

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import BaseModel

# 可能出现在 ResearchState 通道里的类型所在包。
_PROJECT_TYPE_PACKAGES = (
    "deepresearcher.schemas",
    "deepresearcher.evidence",
    "deepresearcher.state",
    "deepresearcher.routing",
    "deepresearcher.reporting",
)


def project_state_types() -> set[tuple[str, ...]]:
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
    return JsonPlusSerializer(allowed_msgpack_modules=project_state_types())
