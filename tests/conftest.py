"""测试运行时与生产 Uvicorn 使用相同的事件循环，OTel provider 全程常驻。"""

import asyncio
import sys

import pytest
from opentelemetry import trace as opentelemetry_trace
from opentelemetry.sdk.trace import TracerProvider

from deepresearcher.observability.tracing import ledger
from pg_backend import postgres_base

try:
    import uvloop
except ImportError:  # pragma: no cover - uvloop 不支持的平台保留标准 asyncio
    uvloop = None


if uvloop is not None:
    # 要在任何 pytest-asyncio fixture 或测试内 asyncio.run() 创建 loop 前设置。
    # Uvicorn[standard] 在 Linux 上也会优先选择 uvloop。
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
elif sys.platform == "win32" and postgres_base() is not None:
    # Windows 默认的 Proactor 循环没有 add_reader，asyncpg 与 psycopg 的异步模式都拒绝在其上
    # 运行。只有配置了 PG 后端才换循环，免得影响无关断言。
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


@pytest.fixture(scope="session", autouse=True)
def otel_tracer_provider():
    """生产 worker_lifespan 会安装 provider，测试会话同样常驻一份，ledger 处理器只挂一次。

    是否记账由 ledger 的 run→sink 注册表决定，未注册 run 的 span 不记账。
    """
    provider = opentelemetry_trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        opentelemetry_trace.set_tracer_provider(TracerProvider())
        provider = opentelemetry_trace.get_tracer_provider()
    assert isinstance(provider, TracerProvider)
    ledger.attach_to_provider(provider)
    return provider
