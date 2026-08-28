"""鉴权：argon2 口令哈希、HS256 JWT 编解码、每请求的 current_user 依赖。

P0 决定：无状态 JWT，不建 sessions 表——代价是**无法服务端吊销**（登出=前端删
token），用短 TTL（默认 12h）兜底。将来加 sessions/jti 黑名单时，改动收敛在
TokenCodec 与 make_current_user 两处。decode 后仍回表查 User：删号立即失效，
且把"这个 id 还存在吗"的检查放在唯一入口里。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import Argon2Error, VerifyMismatchError
from fastapi import HTTPException, Request

from deepsearch_agent.service.models import User

MIN_PASSWORD_CHARS = 8
MAX_PASSWORD_CHARS = 128

_hasher = PasswordHasher()


class TokenError(ValueError):
    """任何 token 层失败（过期/伪造/载荷异常）统一收敛到这里 → 调用方出 401。"""


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except (VerifyMismatchError, Argon2Error, TypeError, ValueError):
        # 当前 argon2 版本里 InvalidHashError 挂在 ValueError 分支而非 Argon2Error；
        # 不区分原因，登录侧一律失败。
        return False


def normalize_email(email: str) -> str:
    return email.strip().lower()


def password_policy_ok(password: str) -> bool:
    return MIN_PASSWORD_CHARS <= len(password) <= MAX_PASSWORD_CHARS


class TokenCodec:
    def __init__(
        self,
        secret: str,
        ttl_hours: int = 12,
        *,
        now: Callable[[], float] | None = None,
    ):
        if len(secret) < 32:
            raise ValueError("JWT secret 长度必须 ≥32 字符。")
        self._secret = secret
        self._ttl_seconds = max(1, ttl_hours) * 3600
        self._now = now or time.time

    def encode(self, user_id: int) -> str:
        issued = int(self._now())
        payload = {"sub": str(user_id), "iat": issued, "exp": issued + self._ttl_seconds}
        return jwt.encode(payload, self._secret, algorithm="HS256")

    def decode(self, token: str) -> int:
        try:
            payload: dict[str, Any] = jwt.decode(
                token, self._secret, algorithms=["HS256"]
            )
            return int(payload["sub"])
        except (jwt.PyJWTError, KeyError, TypeError, ValueError) as exc:
            raise TokenError("无效或已过期的登录凭证。") from exc


def make_current_user(codec: TokenCodec, session_factory: Callable[[], Any]):
    """构造 current_user 依赖：手解 Authorization 头（P0 不依赖 OpenAPI securityScheme，
    换来 lifespan 之后可按 state 组装、无 HTTPBearer 闭包绑定问题）。"""

    async def current_user(request: Request) -> User:
        header = request.headers.get("authorization", "")
        if not header.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="请先登录。")
        try:
            user_id = codec.decode(header[7:].strip())
        except TokenError:
            raise HTTPException(status_code=401, detail="请先登录。") from None
        async with session_factory() as session:
            user = await session.get(User, user_id)
        if user is None:
            raise HTTPException(status_code=401, detail="请先登录。")
        request.state.user = user  # 下游路由复用，避免二次查表
        return user

    return current_user
