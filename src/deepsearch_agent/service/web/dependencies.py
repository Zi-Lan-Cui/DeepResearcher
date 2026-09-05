"""Request-scoped FastAPI dependencies."""

from typing import Any

from fastapi import Depends, HTTPException, Request
from sqlalchemy import select

from deepsearch_agent.service.models import Run, User


def app_state(request: Request) -> Any:
    return request.app.state


async def current_user(request: Request) -> User:
    return await app_state(request).auth_dependency(request)


async def owned_run(
    run_id: str,
    request: Request,
    user: User = Depends(current_user),
) -> Run:
    state = app_state(request)
    async with state.session_factory() as session:
        run = await session.scalar(select(Run).where(Run.id == run_id, Run.user_id == user.id))
    if run is None:
        # Missing and cross-user resources deliberately share one response.
        raise HTTPException(status_code=404, detail="运行不存在。")
    return run
