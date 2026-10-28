"""个人研究偏好端点：读取生效值与可选项、整份保存。"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from deepresearcher.service.preferences import (
    PREFERENCE_CHOICES,
    clamp_preferences,
    frozen_snapshot,
)
from deepresearcher.service.storage.tables import User
from deepresearcher.service.web.dependencies import app_state, current_user
from deepresearcher.service.web.schemas import UpdatePreferencesBody

router = APIRouter(prefix="/api/preferences")


@router.get("")
async def get_preferences(
    request: Request,
    user: User = Depends(current_user),
) -> dict[str, Any]:
    """返回合并后的生效偏好、服务器默认与选项，供账户页表单渲染。"""
    state = app_state(request)
    agent = state.settings.agent
    return {
        "preferences": frozen_snapshot(user.preferences, agent),
        "server_defaults": {
            field_name: getattr(agent, field_name) for field_name in PREFERENCE_CHOICES
        },
        "choices": PREFERENCE_CHOICES,
    }


@router.post("")
async def post_preferences(
    body: UpdatePreferencesBody,
    request: Request,
    user: User = Depends(current_user),
) -> dict[str, Any]:
    """整份提交并保存；非法键值 422。保存后新开的 run 取新值，进行中的不受影响。"""
    state = app_state(request)
    try:
        cleaned = clamp_preferences(body.preferences)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    async with state.session_factory() as session:
        row = await session.get(User, user.id)
        assert row is not None
        row.preferences = cleaned
        await session.commit()
    return {"preferences": cleaned}
