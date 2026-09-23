"""FastAPI application composition root.

Security and transport policy live in ``service.web``. This module deliberately
keeps only application assembly so importing the API does not also hide route,
authentication, SSE, and worker-lifecycle implementations in one large file.
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
    """Build the HTTP control plane and mount its static client.

    The control plane structurally cannot execute runs: graph assembly and the
    worker lifecycle live only in ``deepresearcher.worker``.
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
