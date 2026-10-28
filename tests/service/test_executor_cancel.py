"""执行被取消时的收尾归属：关机让位不写终态，也不吞掉取消。

消费侧按「execute 是否正常返回」决定要不要确认消息，所以这两种取消必须有不同结局。
"""

import asyncio
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from deepresearcher.service.execution import executor as executor_module
from deepresearcher.service.execution.executor import RunExecutor
from deepresearcher.service.runs.store import RunWork, TerminalState

pytestmark = pytest.mark.asyncio

WORK = RunWork(run_id="run-cancel", user_id=1, query="取消收尾", lease_owner="w1", fence_token=3)


class _Buffer:
    def __init__(self) -> None:
        self.done: list[dict] = []
        self.closed: list[str] = []

    def open(self, run_id: str) -> None:
        return None

    def write_done(self, run_id: str, **payload) -> None:
        self.done.append(payload)

    async def flush(self, run_id: str) -> bool:
        return True

    def close(self, run_id: str) -> None:
        self.closed.append(run_id)


class _Harness:
    """只提供 execute 取消分支会碰到的成员，其余走模块级替身。"""

    def __init__(self, *, shutdown: bool) -> None:
        self._buffer = _Buffer()
        self._config = SimpleNamespace(jsonl_events=False)
        self._usage_store = object()
        self._settings = SimpleNamespace(llm=object())
        self._langfuse_client = None
        self._lost_leases: set[str] = set()
        self._shutdown_interrupts = {WORK.run_id} if shutdown else set()
        self.persisted: list[tuple[str, str]] = []

    async def _periodic_flush(self, run_id: str) -> None:
        await asyncio.sleep(3600)

    async def _execute_traced(self, work, *, sink, resume_input):
        raise asyncio.CancelledError

    async def persist_status(self, work, *, status, terminal_reason, **kwargs):
        self.persisted.append((status, terminal_reason))
        return TerminalState(status=status, answer_mode="", report_available=False)

    def mark_lease_lost(self, run_id: str) -> None:
        self._lost_leases.add(run_id)


async def _run_cancelled(harness: _Harness) -> None:
    with (
        patch.object(
            executor_module,
            "ledger",
            SimpleNamespace(attach_run_sink=lambda *args: None, detach_run_sink=lambda *args: None),
        ),
        patch.object(
            executor_module, "spans", SimpleNamespace(trace=lambda *a, **k: nullcontext())
        ),
        patch.object(executor_module, "bind_usage_runtime", lambda runtime: object()),
        patch.object(executor_module, "reset_usage_runtime", lambda token: None),
        patch.object(executor_module, "CompositeSink", lambda *sinks: object()),
    ):
        await RunExecutor.execute(harness, WORK)  # type: ignore[arg-type] - 只提供取消分支用到的成员


async def test_shutdown_cancel_propagates_without_terminal_write():
    """关机取消向上抛：消费侧据此跳过确认，行留给下一个进程重投。"""
    harness = _Harness(shutdown=True)

    with pytest.raises(asyncio.CancelledError):
        await _run_cancelled(harness)

    assert harness.persisted == []
    assert harness._buffer.done == []


async def test_user_cancel_writes_cancelled_state():
    """非关机的取消要写终态：本进程是这条行最后的收尾者。"""
    harness = _Harness(shutdown=False)

    await _run_cancelled(harness)

    assert harness.persisted == [("cancelled", "user_cancelled")]
