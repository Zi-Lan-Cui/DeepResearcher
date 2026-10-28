"""FastAPI 应用装配根：路由表与静态资源。

运行期对象图（engine / manager / codec / limiter）由 bootstrap.lifespan 装配；
路由、鉴权与 SSE 的实现在 service.web，本模块只做装配。
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from deepresearcher.config import Settings
from deepresearcher.observability.logging_config import get_logger
from deepresearcher.service.bootstrap.layers import MessagingPorts
from deepresearcher.service.bootstrap.lifespan import make_lifespan
from deepresearcher.service.settings import ServiceConfig
from deepresearcher.service.web.routes.auth import router as auth_router
from deepresearcher.service.web.routes.events import router as events_router
from deepresearcher.service.web.routes.preferences import router as preferences_router
from deepresearcher.service.web.routes.runs import router as runs_router
from deepresearcher.service.web.routes.stats import router as stats_router

# 三个手写静态文件，无构建步骤，随 wheel 分发，必须以包内相对路径定位。
# 本模块在 bootstrap/ 下，静态资源在 service/web/static，因此上溯一层。
_STATIC_DIR = Path(__file__).resolve().parent.parent / "web" / "static"


def create_app(
    settings: Settings | None = None,
    config: ServiceConfig | None = None,
    *,
    messaging: MessagingPorts | None = None,
) -> FastAPI:
    """装配 HTTP 控制层并挂载静态资源。

    控制层在结构上无法执行 run：图装配与 worker 生命周期
    只存在于 deepresearcher.service.bootstrap.worker。
    """

    app = FastAPI(
        title="DeepResearcher Service",
        lifespan=make_lifespan(settings, config, messaging=messaging),
    )
    app.state.service_logger = get_logger("deepresearcher.service.bootstrap.api")
    app.include_router(auth_router)
    app.include_router(events_router)
    app.include_router(runs_router)
    app.include_router(preferences_router)
    app.include_router(stats_router)
    app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")
    return app
