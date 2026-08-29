"""服务进程入口：``uv run python server.py``。

配置来自 env/.env（SERVICE_* 键）；引擎配置沿用 get_settings()。
"""

import uvicorn

from deepsearch_agent.config import get_settings
from deepsearch_agent.observability import configure_logging
from deepsearch_agent.service.api import create_app
from deepsearch_agent.service.settings import get_service_config


def main() -> None:
    config = get_service_config()
    # 与 CLI 同源：不配置 handler，服务层 INFO/WARNING 会被 logging 的
    # lastResort（仅 WARNING+）静默吞掉——曾导致 text_delta 路由探针无迹可寻。
    settings = get_settings()
    configure_logging(
        settings.app.log_level,
        log_path=settings.observability.log_dir / settings.observability.log_file,
    )
    uvicorn.run(create_app(), host=config.host, port=config.port)


if __name__ == "__main__":
    main()
