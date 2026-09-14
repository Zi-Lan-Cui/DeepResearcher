"""提示词外置层：每个 agent/节点的 system 提示词以纯文本存放在本目录。

动机：提示词会随调优持续变长，混在 Python 里难读难 diff。抽成 `.md` 后，
"改 prompt" 与 "改代码" 分离，且这些文件本身就是缓存前缀里最静态的一层。

约定：
- `load_prompt(name)` 逐字返回 `prompts/<name>.md`（不 strip、不改行尾）——
  调用点靠拼接 `language_directive(...)` / `.replace("__LANG__", …)` 组装，
  与外置前的运行时字符串逐字一致；
- 语言纪律模板在 `language.md`，`{language}` 由 `config.language_directive` 填。
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    """读取 `prompts/<name>.md` 的逐字内容并缓存（网关前缀靠它稳定，避免重复 IO）。"""
    return (resources.files(__package__) / f"{name}.md").read_text("utf-8")


def json_data_section(title: str, payload: Any) -> str:
    """将运行时数据渲染为明确的 Markdown JSON 分区。"""
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    # 用户文本可能自带代码围栏；动态增长围栏，避免数据提前闭合分区。
    fence = "```"
    while fence in rendered:
        fence += "`"
    return f"## {title}\n\n{fence}json\n{rendered}\n{fence}"
