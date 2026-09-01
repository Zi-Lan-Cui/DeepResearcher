"""测试运行时与生产 Uvicorn 使用相同的事件循环。"""

import asyncio

try:
    import uvloop
except ImportError:  # pragma: no cover - uvloop 不支持的平台保留标准 asyncio
    uvloop = None


if uvloop is not None:
    # 要在任何 pytest-asyncio fixture 或测试内 asyncio.run() 创建 loop 前设置。
    # Uvicorn[standard] 在 Linux 上也会优先选择 uvloop。
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
