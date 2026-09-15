"""HTTP request schemas."""

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
