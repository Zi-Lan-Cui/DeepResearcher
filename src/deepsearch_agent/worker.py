"""Independent execution-plane entry: ``python -m deepsearch_agent.worker``."""

from __future__ import annotations

import asyncio
import signal

from deepsearch_agent.config import get_settings
from deepsearch_agent.observability import configure_logging
from deepsearch_agent.service.execution.runtime import worker_lifespan


async def run() -> None:
    """Run until SIGINT/SIGTERM cancels the main task."""
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
