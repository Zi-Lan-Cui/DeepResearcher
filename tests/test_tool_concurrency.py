import asyncio
from types import SimpleNamespace

from deepresearcher.agents.middleware.concurrency import ToolExecutionGate
from deepresearcher.agents.middleware.serial_tools import SerialToolMiddleware


def test_tool_execution_gate_allows_parallel_shared_calls():
    gate = ToolExecutionGate()
    active = 0
    peak = 0

    async def reader():
        nonlocal active, peak
        async with gate.shared():
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0)
            active -= 1

    async def run():
        await asyncio.gather(reader(), reader())

    asyncio.run(run())
    assert peak == 2


def test_tool_execution_gate_makes_exclusive_call_a_barrier():
    gate = ToolExecutionGate()
    entered = asyncio.Event()
    release = asyncio.Event()
    order: list[str] = []

    async def reader():
        async with gate.shared():
            order.append("reader_started")
            entered.set()
            await release.wait()
            order.append("reader_finished")

    async def writer():
        await entered.wait()
        async with gate.exclusive():
            order.append("writer")

    async def run():
        reader_task = asyncio.create_task(reader())
        writer_task = asyncio.create_task(writer())
        await entered.wait()
        await asyncio.sleep(0)
        release.set()
        await asyncio.gather(reader_task, writer_task)

    asyncio.run(run())
    assert order == ["reader_started", "reader_finished", "writer"]


def test_serial_middleware_orders_complete_after_inflight_delegate():
    """生产路径:同批 [ResearchDelegate, ResearchComplete] 交叠时,exclusive 的
    Complete 必须等在飞 delegate(shared)落地——消灭"完结后状态仍可变"竞态。"""

    middleware = SerialToolMiddleware(serial_tools={"ResearchComplete"})
    context = SimpleNamespace(tool_gate=ToolExecutionGate())
    order: list[str] = []
    delegate_running = asyncio.Event()
    delegate_release = asyncio.Event()

    def request(name: str) -> SimpleNamespace:
        return SimpleNamespace(
            runtime=SimpleNamespace(context=context),
            tool_call={"name": name, "id": f"call-{name}", "args": {}},
        )

    async def delegate_handler(_request):
        order.append("delegate_running")
        delegate_running.set()
        await delegate_release.wait()
        order.append("delegate_done")
        return "delegated"

    async def complete_handler(_request):
        order.append("complete")
        return "completed"

    async def run():
        delegate = asyncio.create_task(
            middleware.awrap_tool_call(request("ResearchDelegate"), delegate_handler)
        )
        await delegate_running.wait()
        complete = asyncio.create_task(
            middleware.awrap_tool_call(request("ResearchComplete"), complete_handler)
        )
        await asyncio.sleep(0)
        delegate_release.set()
        await asyncio.gather(delegate, complete)

    asyncio.run(run())
    assert order == ["delegate_running", "delegate_done", "complete"]
