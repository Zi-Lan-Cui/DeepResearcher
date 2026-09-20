"""提示词外置层：每个 agent/节点的 system 提示词以纯文本存放在本目录。

动机：提示词会随调优持续变长，混在 Python 里难读难 diff。抽成 `.md` 后，
"改 prompt" 与 "改代码" 分离，且这些文件本身就是缓存前缀里最静态的一层。

约定：
- `load_prompt(name)` 逐字返回 `prompts/<name>.md`（不 strip、不改行尾）——
  调用点靠拼接 `language_directive(...)` / `.replace("__LANG__", …)` 组装，
  与外置前的运行时字符串逐字一致；
- 语言纪律模板在 `language.md`，`{language}` 由 `config.language_directive` 填；
- `get_runtime_environment()` 提供随每次调用变化的环境事实（当前日期/时区）。
  它是另一种"提示词素材"：静态正文在 .md，动态事实由它渲染进
  `render_data_section("运行时环境", …)`——本包唯一 .py 因此就是门面自身。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import lru_cache
from importlib import resources
from typing import Any


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    """读取 `prompts/<name>.md` 的逐字内容并缓存（网关前缀靠它稳定，避免重复 IO）。"""
    return (resources.files(__package__) / f"{name}.md").read_text("utf-8")


def language_directive(language: str) -> str:
    """生成注入各 agent/node system prompt 的语言纪律行（模板见 `language.md`）。"""
    return load_prompt("language").format(language=language)


def render_data_section(title: str, payload: Any) -> str:
    """将运行时数据渲染为明确的 Markdown JSON 分区。"""
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    # 用户文本可能自带代码围栏；动态增长围栏，避免数据提前闭合分区。
    fence = "```"
    while fence in rendered:
        fence += "`"
    return f"## {title}\n\n{fence}json\n{rendered}\n{fence}"


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
