"""服务进程入口：``uv run python server.py``。

配置来自 env/.env（SERVICE_* 键）；引擎配置沿用 get_settings()。
"""

import uvicorn

from deepsearch_agent.service.api import create_app
from deepsearch_agent.service.settings import get_service_config


def main() -> None:
    config = get_service_config()
    uvicorn.run(create_app(), host=config.host, port=config.port)


if __name__ == "__main__":
    main()
