"""服务进程入口：``uv run python server.py``。

配置来自 env/.env（SERVICE_* 键）；引擎配置沿用 get_settings()。
"""

import uvicorn

from deepresearcher.config import get_settings
from deepresearcher.observability import configure_logging
from deepresearcher.service.api import create_app
from deepresearcher.service.settings import get_service_config


def main() -> None:
    config = get_service_config()
    # 与 Worker 同源:不配置 handler 时,logging 的 lastResort 只输出 WARNING
    # 及以上,服务层的 INFO/WARNING 日志会丢失。
    settings = get_settings()
    configure_logging(
        settings.app.log_level,
        log_path=settings.observability.log_dir / settings.observability.log_file,
    )
    uvicorn.run(
        create_app(),
        host=config.host,
        port=config.port,
        proxy_headers=True,
        forwarded_allow_ips=config.forwarded_allow_ips,
    )


if __name__ == "__main__":
    main()
