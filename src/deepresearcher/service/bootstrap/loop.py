"""进程级引导：API 与 Worker 两个入口共用的事件循环工厂与停止信号安装。"""

from __future__ import annotations

import asyncio
import signal
import sys
from collections.abc import Callable

LoopFactory = Callable[[], asyncio.AbstractEventLoop]


# 不设事件循环策略：Uvicorn 在 win32 上直接把 ProactorEventLoop 交给 asyncio.run，
# 策略无效（uvicorn/loops/asyncio.py）；psycopg 的异步模式没有 add_reader 就连不上库。
# 能生效的只有把零参工厂交给 asyncio.run 或 uvicorn --loop，即下面的工厂函数。
def selector_loop_factory() -> asyncio.AbstractEventLoop:
    """返回 Selector 事件循环，供 asyncio.run(loop_factory=...) 与 uvicorn --loop 使用。

    uvicorn 把 loop 参数写成 module:attribute 时，直接把这个函数当工厂调用，
    因此它不能带参数。
    """
    return asyncio.SelectorEventLoop()


def uvicorn_loop() -> str:
    """交给 uvicorn.run(loop=...) 的取值；非 Windows 保持 auto（Linux 上可用 uvloop）。"""
    return "auto" if sys.platform != "win32" else f"{__name__}:selector_loop_factory"


def asyncio_loop_factory() -> LoopFactory | None:
    """交给 asyncio.run(loop_factory=...) 的取值；None 表示用默认循环。"""
    return None if sys.platform != "win32" else selector_loop_factory


def install_stop_handlers(stopped: asyncio.Event) -> None:
    """把 SIGINT、SIGTERM（Windows 另有 SIGBREAK）接到 stopped。

    必须在主线程调用：signal.signal 只允许主线程注册处理器。
    """
    loop = asyncio.get_running_loop()
    for sig in _stop_signals():
        try:
            loop.add_signal_handler(sig, stopped.set)
        except (NotImplementedError, RuntimeError):
            # Windows 的循环没有 add_signal_handler：处理器在主线程同步执行，
            # 置位要转交给循环线程。
            signal.signal(sig, lambda *_: loop.call_soon_threadsafe(stopped.set))


def _stop_signals() -> tuple[int, ...]:
    signals = [signal.SIGINT, signal.SIGTERM]
    sigbreak = getattr(signal, "SIGBREAK", None)
    if sigbreak is not None:  # pragma: no branch - 仅 Windows 存在
        signals.append(sigbreak)
    return tuple(signals)
