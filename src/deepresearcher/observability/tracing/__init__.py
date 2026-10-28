"""追踪：spans 提供 OTel span 引擎，ledger 把 span 写成事件日志记录。

业务关联标签（run/session/node）不在本包内，统一由 deepresearcher.observability.span
持有 —— 事件模型与 span 记账都要读它，放这里会让 events 与 tracing 互相引用。
"""

from deepresearcher.observability.tracing.spans import span, trace

__all__ = ["span", "trace"]
