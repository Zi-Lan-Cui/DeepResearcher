"""单调计时的共享口径。"""

import time


def elapsed_ms(started_monotonic: float) -> int:
    """从 time.monotonic() 起点到当前的整毫秒数,带负值守卫。"""
    return max(0, round((time.monotonic() - started_monotonic) * 1000))
