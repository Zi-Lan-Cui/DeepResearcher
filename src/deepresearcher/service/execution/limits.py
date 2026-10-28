"""本进程内的两个调用限流器：并发闸门与滑动窗口配额。

两者都只在当前 Worker 进程内生效，不跨副本、不落库，也不感知 run 身份。
"""

from __future__ import annotations

import asyncio
from collections import deque
from time import monotonic

__all__ = ["CapacityGate", "ProviderRateLimiter"]


class CapacityGate:
    """限制同时在途的提供方调用数。"""

    def __init__(self, limit: int) -> None:
        self._semaphore = asyncio.Semaphore(max(1, limit))
        self._active = 0
        self._lock = asyncio.Lock()

    async def acquire(self) -> int:
        await self._semaphore.acquire()
        async with self._lock:
            self._active += 1
            return self._active

    async def release(self) -> None:
        async with self._lock:
            self._active = max(0, self._active - 1)
        self._semaphore.release()


class ProviderRateLimiter:
    """每 Worker 共享的 60 秒滑动窗口 RPM/TPM 限制器。"""

    def __init__(
        self,
        *,
        requests_per_minute: int,
        tokens_per_minute: int,
        window_seconds: float = 60.0,
    ) -> None:
        self._rpm = max(0, requests_per_minute)
        self._tpm = max(0, tokens_per_minute)
        self._window_seconds = max(0.01, window_seconds)
        self._requests: deque[float] = deque()
        self._tokens: deque[tuple[float, int]] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self, estimated_tokens: int) -> None:
        if not self._rpm and not self._tpm:
            return
        reservation = min(max(1, estimated_tokens), self._tpm) if self._tpm else 0
        while True:
            async with self._lock:
                now = monotonic()
                cutoff = now - self._window_seconds
                while self._requests and self._requests[0] <= cutoff:
                    self._requests.popleft()
                while self._tokens and self._tokens[0][0] <= cutoff:
                    self._tokens.popleft()
                request_ok = not self._rpm or len(self._requests) < self._rpm
                token_ok = (
                    not self._tpm
                    or sum(value for _, value in self._tokens) + reservation <= self._tpm
                )
                if request_ok and token_ok:
                    self._requests.append(now)
                    if self._tpm:
                        self._tokens.append((now, reservation))
                    return
                deadlines: list[float] = []
                if not request_ok and self._requests:
                    deadlines.append(self._requests[0] + self._window_seconds)
                if not token_ok and self._tokens:
                    deadlines.append(self._tokens[0][0] + self._window_seconds)
                delay = max(0.01, min(deadlines) - now) if deadlines else 0.01
            await asyncio.sleep(delay)
