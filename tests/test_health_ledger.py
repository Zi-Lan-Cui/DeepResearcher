"""依赖健康账本：连续计数、冷却递增与惰性过期。"""

import pytest

from deepresearcher.health import Cooldown, HealthLedger, MemoryHealthLedger, Unavailable

pytestmark = pytest.mark.asyncio

POLICY = Cooldown(threshold=3, base_seconds=60.0, max_seconds=150.0)


class FakeClock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _ledger() -> tuple[MemoryHealthLedger, FakeClock]:
    clock = FakeClock()
    return MemoryHealthLedger(clock=clock), clock


async def test_unknown_key_is_available():
    ledger, _ = _ledger()
    assert await ledger.unavailable("never-seen") is None


async def test_failures_below_threshold_keep_key_available():
    ledger, _ = _ledger()
    for _ in range(2):
        await ledger.record_failure("search:aliyun", "tool_request", POLICY)
    assert await ledger.unavailable("search:aliyun") is None


async def test_threshold_opens_for_base_duration():
    ledger, clock = _ledger()
    for _ in range(3):
        await ledger.record_failure("search:aliyun", "tool_request", POLICY)
    state = await ledger.unavailable("search:aliyun")
    assert isinstance(state, Unavailable)
    assert state.reason == "tool_request"
    assert state.retry_after == pytest.approx(60.0)
    clock.now += 61
    assert await ledger.unavailable("search:aliyun") is None


async def test_success_resets_consecutive_count():
    ledger, _ = _ledger()
    await ledger.record_failure("search:aliyun", "tool_request", POLICY)
    await ledger.record_failure("search:aliyun", "tool_request", POLICY)
    await ledger.record_success("search:aliyun")
    await ledger.record_failure("search:aliyun", "tool_request", POLICY)
    assert await ledger.unavailable("search:aliyun") is None


async def test_repeated_ejection_escalates_then_caps():
    ledger, clock = _ledger()
    durations: list[float] = []
    for _ in range(4):
        for _ in range(3):
            await ledger.record_failure("search:aliyun", "tool_request", POLICY)
        state = await ledger.unavailable("search:aliyun")
        assert state is not None
        durations.append(state.retry_after)
        clock.now += state.retry_after + 1
    assert durations == pytest.approx([60.0, 120.0, 150.0, 150.0])


async def test_trip_is_immediate_and_not_counted_as_strike():
    ledger, clock = _ledger()
    await ledger.trip("search:tavily", "quota_exhausted", 900)
    state = await ledger.unavailable("search:tavily")
    assert state is not None
    assert state.retry_after == pytest.approx(900.0)
    clock.now += 901
    for _ in range(3):
        await ledger.record_failure("search:tavily", "tool_request", POLICY)
    state = await ledger.unavailable("search:tavily")
    assert state is not None
    assert state.retry_after == pytest.approx(60.0)


async def test_trip_never_shortens_active_window():
    ledger, _ = _ledger()
    await ledger.trip("search:tavily", "quota_exhausted", 900)
    await ledger.trip("search:tavily", "rate_limited", 30)
    state = await ledger.unavailable("search:tavily")
    assert state is not None
    assert state.retry_after == pytest.approx(900.0)


async def test_memory_ledger_satisfies_protocol():
    assert isinstance(MemoryHealthLedger(), HealthLedger)
