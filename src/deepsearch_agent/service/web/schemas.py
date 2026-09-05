"""HTTP request schemas."""

from pydantic import BaseModel, Field

MAX_QUERY_CHARS = 2_000


class RegisterBody(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=256)


class LoginBody(BaseModel):
    email: str
    password: str


class CreateRunBody(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)


class ResumeRunBody(BaseModel):
    answer: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
