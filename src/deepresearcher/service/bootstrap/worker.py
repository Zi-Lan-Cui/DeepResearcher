"""独立执行入口：python -m deepresearcher.service.bootstrap.worker。"""

from __future__ import annotations

import asyncio

from deepresearcher.config import get_settings
from deepresearcher.observability import configure_logging
from deepresearcher.service.bootstrap.loop import asyncio_loop_factory, install_stop_handlers
from deepresearcher.service.execution.runtime import worker_lifespan


async def run() -> None:
    """运行到 SIGINT/SIGTERM 取消主任务为止。"""
    stopped = asyncio.Event()
    install_stop_handlers(stopped)
    async with worker_lifespan():
        await stopped.wait()


def main() -> None:
    settings = get_settings()
    configure_logging(
        settings.app.log_level,
        log_path=settings.observability.log_dir / settings.observability.log_file,
    )
    try:
        # 循环必须在建连接池之前就换掉，见 service.bootstrap.loop.selector_loop_factory。
        asyncio.run(run(), loop_factory=asyncio_loop_factory())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
