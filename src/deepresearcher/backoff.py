"""统一的退避逻辑实现。"""

import random

# 指数曲线的倍增底数：本地曲线与模型重试中间件共用，避免各写一个 2.0。
DEFAULT_BACKOFF_FACTOR = 2.0


def delay_seconds(
    attempt: int,
    *,
    initial_delay: float,
    max_delay: float,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    jitter_ratio: float = 0.0,
    retry_after: float | None = None,
    retry_after_cap: float | None = None,
) -> float:
    """第 attempt 次重试前应等待的秒数，attempt 从 0 计。

    服务端声明等待时长（retry_after）优先于本地曲线，且只受 retry_after_cap
    约束，不看 max_delay。jitter_ratio 在结果上叠加正向随机量，错开并发
    请求的重试时点；为零 0 则不引入随机。
    """
    if retry_after is not None:
        base = retry_after if retry_after_cap is None else min(retry_after, retry_after_cap)
    else:
        base = min(max_delay, initial_delay * (backoff_factor**attempt))
    base = max(0.0, base)
    if jitter_ratio > 0:
        base += random.uniform(0.0, base * jitter_ratio)
    return base
