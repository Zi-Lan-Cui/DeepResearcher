"""环境变量读取的解析语义单源:strip、大小写、失败回退只在这一处定义。

引擎 Settings(config.py)与服务 ServiceConfig(settings.py)两套配置共享本模块;
各自的校验/夹取(max、choices、必填)留在自己的构造处,这里只保证
"同一种写法在两边解析结果相同"。
"""

from __future__ import annotations

from os import getenv


def env_str(name: str, default: str = "") -> str:
    return getenv(name, default).strip()


def env_int(name: str, default: int) -> int:
    try:
        return int(env_str(name, str(default)))
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(env_str(name, str(default)))
    except ValueError:
        return default


def env_bool(name: str, default: bool) -> bool:
    raw = env_str(name, "true" if default else "false").lower()
    if raw in {"1", "true", "yes"}:
        return True
    if raw in {"0", "false", "no"}:
        return False
    return default
