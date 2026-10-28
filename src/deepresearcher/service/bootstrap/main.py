"""API 进程入口：console script `deepresearcher-api`，即 `python -m deepresearcher.service.bootstrap.main`。"""

import uvicorn

from deepresearcher.config import get_settings
from deepresearcher.observability import configure_logging
from deepresearcher.service.bootstrap.api import create_app
from deepresearcher.service.bootstrap.loop import uvicorn_loop
from deepresearcher.service.settings import get_service_config


def main() -> None:
    config = get_service_config()
    settings = get_settings()
    # 配置日志输出位置
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
        loop=uvicorn_loop(),
    )


if __name__ == "__main__":
    main()
