"""FastAPI 应用装配：鉴权、运行受理、SSE 事件流、静态前端。

安全边界（P0 检查单）：
- 每个 /api/runs* 查询都带 user_id 条件；越权一律 404（不暴露他人资源存在性）；
- 事件出站前必经 projector 白名单投影，DB 回放路径复用同一投影器；
- 登录失败统一 401 文案，不区分“无此人/密码错”；
- 同源 StaticFiles 托管前端 → 无需 CORS；若将来前端独立部署再补。
显式不做（见 docs/service-p0.md）：登录限流、token 吊销、SSRF 守卫（P2）。
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from deepsearch_agent.config import Settings, get_settings
from deepsearch_agent.orchestration.graph import build_graph
from deepsearch_agent.service.auth import (
    TokenCodec,
    hash_password,
    make_current_user,
    normalize_email,
    password_policy_ok,
    verify_password,
)
from deepsearch_agent.service.db import init_db, make_engine, make_session_factory
from deepsearch_agent.service.events import CLOSE_STREAM, FanoutSink
from deepsearch_agent.service.models import Run, RunEvent, User
from deepsearch_agent.service.projector import project
from deepsearch_agent.service.runs import QuotaExceededError, RunManager
from deepsearch_agent.service.settings import (
    ServiceConfig,
    checkpoint_dsn,
    get_service_config,
)
from deepsearch_agent.tools.transport import HttpClient

_FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
_SSE_HEARTBEAT_SECONDS = 15.0
_MAX_QUERY_CHARS = 2000


class RegisterBody(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=256)


class LoginBody(BaseModel):
    email: str
    password: str


class CreateRunBody(BaseModel):
    query: str = Field(min_length=1, max_length=_MAX_QUERY_CHARS)


def _run_summary(run: Run) -> dict[str, Any]:
    return {
        "id": run.id,
        "query": run.query,
        "status": run.status,
        "answer_mode": run.answer_mode,
        "terminal_reason": run.terminal_reason,
        "evidence_count": run.evidence_count,
        "source_count": run.source_count,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
    }


def _get_state(request: Request) -> Any:
    return request.app.state


def create_app(
    settings: Settings | None = None,
    config: ServiceConfig | None = None,
    *,
    graph_factory: Callable[..., Any] = build_graph,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        cfg = config or get_service_config()
        engine_settings = settings or get_settings()
        engine = make_engine(cfg.database_url)
        await init_db(engine)
        session_factory = make_session_factory(engine)
        # HttpClient 进程级一个：跨 run 共享 provider 信号量与断路器（图不会关闭外来的它）。
        http_client = HttpClient(engine_settings.search)
        fanout = FanoutSink(asyncio.get_running_loop())

        # checkpointer：仅 PostgreSQL 部署启用（AsyncPostgresSaver 走 psycopg，与
        # 业务库同实例、独立连接）。SQLite 测试路径 checkpoint_dsn 返回 None，
        # 恢复能力随部署形态自动降级——不假装有。
        checkpoint_cm = None
        checkpointer = None
        dsn = checkpoint_dsn(cfg.database_url)
        if dsn is not None:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            checkpoint_cm = AsyncPostgresSaver.from_conn_string(dsn)
            checkpointer = await checkpoint_cm.__aenter__()
            await checkpointer.setup()

        manager = RunManager(
            settings=engine_settings,
            session_factory=session_factory,
            config=cfg,
            fanout=fanout,
            http_client=http_client,
            graph_factory=graph_factory,
            checkpointer=checkpointer,
        )
        recovered = await manager.reconcile_startup()
        if recovered:
            app.state.service_logger.info("reconciled_stale_runs count=%d", recovered)

        app.state.config = cfg
        app.state.settings = engine_settings
        app.state.engine = engine
        app.state.session_factory = session_factory
        app.state.fanout = fanout
        app.state.manager = manager
        app.state.checkpointer = checkpointer
        app.state.codec = TokenCodec(cfg.jwt_secret, cfg.token_ttl_hours)
        app.state.auth_dependency = make_current_user(app.state.codec, session_factory)
        try:
            yield
        finally:
            await manager.shutdown()
            await http_client.aclose()
            if checkpoint_cm is not None:
                await checkpoint_cm.__aexit__(None, None, None)
            await engine.dispose()

    from deepsearch_agent.observability.logger import get_logger

    app = FastAPI(title="deepsearch-agent service", lifespan=lifespan)
    app.state.service_logger = get_logger("deepsearch_agent.service.api")

    # Depends 在路由注册期就要求固定 callable，而真正的鉴权闭包要到 lifespan
    # （codec/session_factory 就绪）才能组装——用一层从 request 现取 state 的代理。
    async def auth_dependency(request: Request) -> User:
        return await _get_state(request).auth_dependency(request)

    @app.post("/api/register", status_code=201)
    async def register(body: RegisterBody, request: Request) -> JSONResponse:
        email = normalize_email(body.email)
        if not password_policy_ok(body.password):
            raise HTTPException(status_code=422, detail="密码长度须在 8-128 字符之间。")
        state = _get_state(request)
        async with state.session_factory() as session:
            existing = await session.scalar(select(User).where(User.email == email))
            if existing is not None:
                raise HTTPException(status_code=409, detail="该邮箱已注册。")
            user = User(email=email, password_hash=hash_password(body.password))
            session.add(user)
            try:
                await session.commit()
            except IntegrityError:
                # 查重只是快速路径，真正守门的是 UNIQUE 索引：并发的第二个
                # 同名注册会走到这里（不修则 500）。配额那种跨行不变量才需要
                # 应用锁；单列唯一交给数据库，我们只负责把冲突翻成 409。
                raise HTTPException(status_code=409, detail="该邮箱已注册。") from None
            token = state.codec.encode(user.id)
            return JSONResponse(
                status_code=201,
                content={"token": token, "user": {"id": user.id, "email": user.email}},
            )

    @app.post("/api/login")
    async def login(body: LoginBody, request: Request) -> dict[str, Any]:
        email = normalize_email(body.email)
        state = _get_state(request)
        async with state.session_factory() as session:
            user = await session.scalar(select(User).where(User.email == email))
            # 统一 401：不泄露“该邮箱是否注册过”。
            if user is None or not verify_password(body.password, user.password_hash):
                raise HTTPException(status_code=401, detail="邮箱或密码不正确。")
            return {
                "token": state.codec.encode(user.id),
                "user": {"id": user.id, "email": user.email},
            }

    @app.get("/api/me")
    async def me(user: User = Depends(auth_dependency)) -> dict[str, Any]:
        return {
            "id": user.id,
            "email": user.email,
            "created_at": user.created_at.isoformat() if user.created_at else None,
        }

    @app.post("/api/runs", status_code=202)
    async def create_run(
        body: CreateRunBody, request: Request, user: User = Depends(auth_dependency)
    ) -> dict[str, str]:
        query = body.query.strip()
        if not query:
            raise HTTPException(status_code=422, detail="研究问题不能为空。")
        state = _get_state(request)
        try:
            run_id = await state.manager.start(user.id, query)
        except QuotaExceededError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from None
        return {"run_id": run_id}

    @app.get("/api/runs")
    async def list_runs(
        request: Request, user: User = Depends(auth_dependency)
    ) -> list[dict[str, Any]]:
        state = _get_state(request)
        async with state.session_factory() as session:
            runs = (
                await session.scalars(
                    select(Run)
                    .where(Run.user_id == user.id)
                    .order_by(Run.created_at.desc())
                    .limit(50)
                )
            ).all()
        return [_run_summary(run) for run in runs]

    async def _owned_run(request: Request, user: User, run_id: str) -> Run:
        state = _get_state(request)
        async with state.session_factory() as session:
            run = await session.scalar(
                select(Run).where(Run.id == run_id, Run.user_id == user.id)
            )
        if run is None:  # 越权与不存在同码：不暴露他人 run 的存在性
            raise HTTPException(status_code=404, detail="运行不存在。")
        return run

    @app.get("/api/runs/{run_id}")
    async def get_run(
        run_id: str, request: Request, user: User = Depends(auth_dependency)
    ) -> dict[str, Any]:
        run = await _owned_run(request, user, run_id)
        return {
            **_run_summary(run),
            "report_markdown": run.report_markdown,
            "citations": run.citations_json or [],
            "error_message": run.error_message,
        }

    @app.post("/api/runs/{run_id}/cancel")
    async def cancel_run(
        run_id: str, request: Request, user: User = Depends(auth_dependency)
    ) -> dict[str, str]:
        state = _get_state(request)
        await _owned_run(request, user, run_id)  # 先拥有者校验
        try:
            run = await state.manager.cancel(user.id, run_id)
        except LookupError:
            raise HTTPException(status_code=404, detail="运行不存在。") from None
        return {"id": run.id, "status": run.status}

    @app.get("/api/runs/{run_id}/events")
    async def run_events(
        run_id: str, request: Request, user: User = Depends(auth_dependency)
    ) -> StreamingResponse:
        await _owned_run(request, user, run_id)
        state = _get_state(request)

        async def stream() -> AsyncIterator[str]:
            # ① subscribe 先行（含 backlog），② DB 回放旧段，③ 队列续活段——
            # 三段统一按 seq 去重，任何时刻接入都不重不漏。
            key, queue = state.fanout.subscribe(run_id)
            last_seq = 0
            try:
                async with state.session_factory() as session:
                    rows = (
                        await session.scalars(
                            select(RunEvent)
                            .where(RunEvent.run_id == run_id)
                            .order_by(RunEvent.seq)
                        )
                    ).all()
                for row in rows:
                    if row.seq <= last_seq:
                        continue
                    last_seq = row.seq
                    frame = project(row.record)
                    if frame is not None:
                        yield _sse(frame)
                        if frame.event == "done":
                            return
                while True:
                    try:
                        item = await asyncio.wait_for(queue.get(), _SSE_HEARTBEAT_SECONDS)
                    except TimeoutError:
                        yield ": ping\n\n"  # SSE 注释帧：保活，客户端自动忽略
                        continue
                    if item is CLOSE_STREAM:
                        return
                    # ephemeral 预览帧无 seq：只走直播通道，不参与回放/去重。
                    if item.get("event_type") == "text_delta":
                        frame = project(item)
                        if frame is not None:
                            yield _sse(frame)
                        continue
                    seq = item.get("seq")
                    if not isinstance(seq, int) or seq <= last_seq:
                        continue
                    last_seq = seq
                    frame = project(item)
                    if frame is not None:
                        yield _sse(frame)
                        if frame.event == "done":
                            return
            finally:
                state.fanout.unsubscribe(run_id, key)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # API 路由先注册、静态兜底后挂：/api/* 永远优先于 SPA 回退。
    app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")
    return app


def _sse(frame: Any) -> str:
    data = json.dumps(frame.data, ensure_ascii=False, default=str)
    return f"id: {frame.data.get('seq', 0)}\nevent: {frame.event}\ndata: {data}\n\n"
