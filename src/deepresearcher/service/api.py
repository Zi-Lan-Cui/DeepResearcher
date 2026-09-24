"""FastAPI 应用装配根。

安全与传输策略住在 ``service.web``。本模块刻意只保留应用装配，
避免导入 API 时把路由、鉴权、SSE 与 worker 生命周期实现
藏进同一个大文件。
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from deepresearcher.config import Settings
from deepresearcher.observability.logging_config import get_logger
from deepresearcher.service.settings import ServiceConfig
from deepresearcher.service.web.routes.auth import router as auth_router
from deepresearcher.service.web.routes.events import router as events_router
from deepresearcher.service.web.routes.runs import router as runs_router
from deepresearcher.service.web.runtime import make_lifespan

_FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"


def create_app(
    settings: Settings | None = None,
    config: ServiceConfig | None = None,
) -> FastAPI:
    """装配 HTTP 控制面并挂载静态前端。

    控制面在结构上无法执行 run：图装配与 worker 生命周期
    只存在于 ``deepresearcher.worker``。

    返回:
        FastAPI: 配置好 lifespan 的应用实例。
    """

    app = FastAPI(
        title="DeepResearcher Service",
        lifespan=make_lifespan(settings, config),
    )
    app.state.service_logger = get_logger("deepresearcher.service.api")
    app.include_router(auth_router)
    app.include_router(events_router)
    app.include_router(runs_router)
    app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")
    return app
