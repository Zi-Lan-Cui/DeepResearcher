"""退避时长的计算口径：曲线、上限、服务端指示优先。"""

import pytest

from deepresearcher import backoff
from deepresearcher.backoff import delay_seconds


def test_exponential_curve_from_zero_attempt():
    """尝试次数从 0 计：首次重试等待 initial_delay，此后按底数倍增。"""
    delays = [delay_seconds(attempt, initial_delay=0.5, max_delay=100.0) for attempt in range(4)]

    assert delays == [0.5, 1.0, 2.0, 4.0]


def test_curve_is_capped_by_max_delay():
    """本地曲线受 max_delay 封顶，长时间失败不会把等待拉成无限。"""
    assert delay_seconds(10, initial_delay=1.0, max_delay=8.0) == 8.0


def test_retry_after_wins_over_curve_and_ignores_max_delay():
    """服务端给出的等待时长优先，且不被本地 max_delay 截断。"""
    assert delay_seconds(0, initial_delay=1.0, max_delay=5.0, retry_after=30.0) == 30.0


def test_retry_after_is_capped_by_its_own_limit():
    """服务端要求过长等待时按 retry_after_cap 单独截断。"""
    assert (
        delay_seconds(0, initial_delay=1.0, max_delay=5.0, retry_after=3600.0, retry_after_cap=45.0)
        == 45.0
    )


def test_negative_inputs_yield_zero():
    """负等待无意义，按 0 处理而不是让 sleep 抛错。"""
    assert delay_seconds(0, initial_delay=0.0, max_delay=0.0) == 0.0
    assert delay_seconds(0, initial_delay=1.0, max_delay=5.0, retry_after=-3.0) == 0.0


def test_jitter_within_ratio(monkeypatch):
    """抖动只在基准之上叠加，落在 [base, base * (1 + ratio)] 内。"""
    seen: list[float] = []

    def fake_uniform(low: float, high: float) -> float:
        seen.append(high)
        return high

    monkeypatch.setattr(backoff.random, "uniform", fake_uniform)

    assert delay_seconds(0, initial_delay=4.0, max_delay=10.0, jitter_ratio=0.25) == 5.0
    assert seen == [1.0]


def test_no_random_draw_without_jitter(monkeypatch):
    """jitter_ratio 为 0 时不引入随机，结果精确等于基准。"""

    def fail(low: float, high: float) -> float:
        raise AssertionError("不应取随机数")

    monkeypatch.setattr(backoff.random, "uniform", fail)

    assert delay_seconds(2, initial_delay=1.0, max_delay=100.0) == 4.0


@pytest.mark.parametrize("factor", [2.0, 1.5])
def test_custom_factor(factor):
    """底数可替换，不影响封顶逻辑。"""
    assert delay_seconds(3, initial_delay=1.0, max_delay=100.0, backoff_factor=factor) == factor**3
