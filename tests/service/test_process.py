"""进程引导的守卫：Windows 上不能落到 Proactor 循环，也不能缺少停止信号。"""

from __future__ import annotations

import asyncio
import os
import signal
import sys

import pytest

from deepresearcher.service.bootstrap import loop as process_loop

WINDOWS = sys.platform == "win32"


def _stop_signals() -> list[int]:
    signals = [signal.SIGINT, signal.SIGTERM]
    sigbreak = getattr(signal, "SIGBREAK", None)
    if sigbreak is not None:
        signals.append(sigbreak)
    return signals


def test_selector_loop_factory_returns_a_usable_loop():
    loop = process_loop.selector_loop_factory()
    try:
        assert isinstance(loop, asyncio.AbstractEventLoop)
        assert not loop.is_closed()
    finally:
        loop.close()


def test_windows_gets_the_selector_loop_and_others_keep_defaults():
    if WINDOWS:
        assert process_loop.asyncio_loop_factory() is process_loop.selector_loop_factory
        assert (
            process_loop.uvicorn_loop()
            == "deepresearcher.service.bootstrap.loop:selector_loop_factory"
        )
    else:
        assert process_loop.asyncio_loop_factory() is None
        assert process_loop.uvicorn_loop() == "auto"


@pytest.mark.asyncio
async def test_stop_signal_sets_the_event_on_every_platform():
    stopped = asyncio.Event()
    original = {sig: signal.getsignal(sig) for sig in _stop_signals()}
    try:
        process_loop.install_stop_handlers(stopped)
        # 无论走 add_signal_handler 还是 signal.signal，默认处理器都必须被换掉。
        assert signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL
        if WINDOWS:
            # Windows 的处理器由标准库注册，手工调一次等价于内核投递 CTRL_BREAK。
            handler = signal.getsignal(signal.SIGTERM)
            assert callable(handler)
            handler(signal.SIGTERM, None)
            await asyncio.sleep(0)
        else:
            os.kill(os.getpid(), signal.SIGTERM)
            await asyncio.sleep(0.05)
        assert stopped.is_set()
    finally:
        for sig, handler in original.items():
            try:
                signal.signal(sig, handler)
            except (OSError, ValueError):  # pragma: no cover - 非主线程
                pass
