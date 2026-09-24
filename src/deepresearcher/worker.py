"""独立执行平面入口：``python -m deepresearcher.worker``。"""

from __future__ import annotations

import asyncio
import signal

from deepresearcher.config import get_settings
from deepresearcher.observability import configure_logging
from deepresearcher.service.execution.runtime import worker_lifespan


async def run() -> None:
    """运行到 SIGINT/SIGTERM 取消主任务为止。"""
    stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stopped.set)
        except NotImplementedError:  # pragma: no cover - Windows event loop
            pass
    async with worker_lifespan():
        await stopped.wait()


def main() -> None:
    settings = get_settings()
    configure_logging(
        settings.app.log_level,
        log_path=settings.observability.log_dir / settings.observability.log_file,
    )
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
