"""Web 服务层。

只允许 service → 引擎 的单向依赖；引擎不得 import 本包。
配置入口为 :func:`get_service_config`，与引擎 ``get_settings()`` 读取同一份 env 文件。
"""

from deepresearcher.service.settings import (
    ServiceConfig,
    clear_service_config_cache,
    get_service_config,
)

__all__ = ["ServiceConfig", "clear_service_config_cache", "get_service_config"]
