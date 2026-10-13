"""注册、登录与当前用户路由。"""

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from deepresearcher.service.auth import (
    hash_password,
    normalize_email,
    password_policy_ok,
    verify_password,
)
from deepresearcher.service.persistence.models import Run, RunUsage, User
from deepresearcher.service.runs.transitions import OUTSTANDING_STATUSES
from deepresearcher.service.web.dependencies import app_state, current_user
from deepresearcher.service.web.schemas import LoginBody, RegisterBody

router = APIRouter(prefix="/api")


@router.post("/register", status_code=201)
async def register(body: RegisterBody, request: Request) -> JSONResponse:
    email = normalize_email(body.email)
    if not password_policy_ok(body.password):
        raise HTTPException(status_code=422, detail="密码长度须在 8-128 字符之间。")
    state = app_state(request)
    async with state.session_factory() as session:
        existing = await session.scalar(select(User).where(User.email == email))
        if existing is not None:
            raise HTTPException(status_code=409, detail="该邮箱已注册。")
        user = User(email=email, password_hash=hash_password(body.password))
        session.add(user)
        try:
            await session.commit()
        except IntegrityError:
            raise HTTPException(status_code=409, detail="该邮箱已注册。") from None
        token = state.codec.encode(user.id)
        return JSONResponse(
            status_code=201,
            content={"token": token, "user": {"id": user.id, "email": user.email}},
        )


@router.post("/login")
async def login(body: LoginBody, request: Request) -> dict[str, Any]:
    email = normalize_email(body.email)
    state = app_state(request)
    client_ip = request.client.host if request.client is not None else "unknown"
    admission = await state.login_rate_limiter.consume(client_ip=client_ip, email=email)
    if not admission.allowed:
        raise HTTPException(
            status_code=429,
            detail="登录尝试过于频繁，请稍后再试。",
            headers={"Retry-After": str(admission.retry_after_seconds)},
        )
    async with state.session_factory() as session:
        user = await session.scalar(select(User).where(User.email == email))
        if user is None or not verify_password(body.password, user.password_hash):
            raise HTTPException(status_code=401, detail="邮箱或密码不正确。")
        user_id, user_email = user.id, user.email
    await state.login_rate_limiter.clear_account(email)
    return {
        "token": state.codec.encode(user_id),
        "user": {"id": user_id, "email": user_email},
    }


@router.get("/me")
async def me(request: Request, user: User = Depends(current_user)) -> dict[str, Any]:
    """当前用户的身份与账本视图。"""
    state = app_state(request)
    day_start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    async with state.session_factory() as session:
        total_runs = await session.scalar(
            select(func.count()).select_from(Run).where(Run.user_id == user.id)
        )
        active_runs = await session.scalar(
            select(func.count())
            .select_from(Run)
            .where(Run.user_id == user.id, Run.status.in_(OUTSTANDING_STATUSES))
        )
        cost_today = await session.scalar(
            select(func.coalesce(func.sum(RunUsage.cost_cny), 0))
            .join(Run, RunUsage.run_id == Run.id)
            .where(
                Run.user_id == user.id,
                RunUsage.category == "llm",
                RunUsage.created_at >= day_start,
            )
        )
    budgets = state.settings.llm
    return {
        "id": user.id,
        "email": user.email,
        "created_at": user.created_at.isoformat() if user.created_at else None,
        "runs": {"total": total_runs or 0, "active": active_runs or 0},
        "usage_today": {"cost_cny": float(cost_today or 0)},
        "quotas": {
            "max_concurrent_runs_per_user": state.config.max_concurrent_runs_per_user,
            "max_tokens_per_run": budgets.max_tokens_per_run,
            "max_cost_cny_per_run": budgets.max_cost_cny_per_run,
            "max_cost_cny_per_user_daily": budgets.max_cost_cny_per_user_daily,
        },
    }
