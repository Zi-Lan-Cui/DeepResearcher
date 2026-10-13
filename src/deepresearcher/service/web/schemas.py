"""HTTP 请求模型。"""

from typing import Any

from pydantic import BaseModel, Field

EMAIL_ADDRESS_MAX_CHARS = 320
PASSWORD_INPUT_MAX_CHARS = 256

MAX_QUERY_CHARS = 2_000


class RegisterBody(BaseModel):
    email: str = Field(min_length=3, max_length=EMAIL_ADDRESS_MAX_CHARS)
    password: str = Field(min_length=1, max_length=PASSWORD_INPUT_MAX_CHARS)


class LoginBody(BaseModel):
    email: str = Field(min_length=3, max_length=EMAIL_ADDRESS_MAX_CHARS)
    password: str = Field(min_length=1, max_length=PASSWORD_INPUT_MAX_CHARS)


class CreateRunBody(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)


class ResumeRunBody(BaseModel):
    answer: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)


class UpdatePreferencesBody(BaseModel):
    """整份个人研究偏好提交;键与取值的合法性由 service.preferences 单一事实源校验。"""

    preferences: dict[str, Any]
