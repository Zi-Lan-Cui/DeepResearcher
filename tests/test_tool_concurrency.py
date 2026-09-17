import asyncio

from deepresearcher.context.concurrency import ToolExecutionGate


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
