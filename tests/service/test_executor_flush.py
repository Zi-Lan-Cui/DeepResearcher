"""RunExecutor 周期性 flush 的退避行为：库不可用时不按稳态周期重试。"""

import asyncio

import pytest

from deepresearcher.service.execution import executor as executor_module
from deepresearcher.service.execution.executor import RunExecutor

pytestmark = pytest.mark.asyncio


class _ScriptedBuffer:
    """按预设序列返回 flush 结果，末项无限重复；记录调用次数。"""

    def __init__(self, results: list[bool]) -> None:
        self._results = results
        self.calls = 0

    async def flush(self, run_id: str) -> bool:
        result = self._results[min(self.calls, len(self._results) - 1)]
        self.calls += 1
        return result


class _Harness:
    def __init__(self, buffer: _ScriptedBuffer) -> None:
        self._buffer = buffer


async def _run_cycles(monkeypatch, results: list[bool], cycles: int) -> list[float]:
    """跑指定轮次后以取消收尾，返回每轮实际睡眠时长。"""
    delays: list[float] = []
    real_sleep = asyncio.sleep

    async def _record(delay: float) -> None:
        delays.append(delay)
        if len(delays) >= cycles:
            raise asyncio.CancelledError
        await real_sleep(0)

    monkeypatch.setattr(executor_module.asyncio, "sleep", _record)
    await RunExecutor._periodic_flush(_Harness(_ScriptedBuffer(results)), "run-1")  # noqa: SLF001
    return delays


async def test_flush_failure_backs_off_and_recovers(monkeypatch):
    """连续失败按倍增拉开间隔，一旦写成功立即回到稳态周期。"""
    interval = executor_module._FLUSH_INTERVAL_SECONDS  # noqa: SLF001

    delays = await _run_cycles(monkeypatch, [True, False, False, False, True, True], cycles=6)

    assert delays == [interval, interval, interval * 2, interval * 4, interval * 8, interval]


async def test_flush_backoff_is_capped(monkeypatch):
    """退避有上限：库长时间不可用时重试间隔不应无限拉长。"""
    interval = executor_module._FLUSH_INTERVAL_SECONDS  # noqa: SLF001
    ceiling = executor_module._FLUSH_BACKOFF_MAX_SECONDS  # noqa: SLF001

    delays = await _run_cycles(monkeypatch, [False], cycles=7)

    assert delays[:5] == [interval, interval * 2, interval * 4, interval * 8, interval * 16]
    assert delays[5:] == [ceiling, ceiling]
