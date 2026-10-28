"""分派消息契约：消息队列载荷。"""

from __future__ import annotations

import json
from typing import Final

from pydantic import BaseModel

DISPATCH_SCHEMA_VERSION: Final = 1


class DispatchMessage(BaseModel):
    """一条分派意图：目标 run 与载荷版本。"""

    run_id: str
    schema_version: int = DISPATCH_SCHEMA_VERSION


def new_dispatch(run_id: str) -> DispatchMessage:
    """为一次「让这个 run 被执行」生成消息体。"""
    return DispatchMessage(run_id=run_id)


def encode_dispatch(message: DispatchMessage) -> bytes:
    """把消息序列化成投递载荷。"""
    return json.dumps(message.model_dump(), ensure_ascii=False).encode("utf-8")


def decode_dispatch(body: bytes) -> DispatchMessage:
    """解析消息体；载荷不合法或版本高于当前时抛 ValueError。

    调用方把这个异常当作毒消息处理：内容不会因为重投而变好。
    """
    try:
        raw = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"分派消息不是合法 JSON：{exc}") from exc
    message = DispatchMessage.model_validate(raw)
    if message.schema_version > DISPATCH_SCHEMA_VERSION:
        raise ValueError(f"不认识的分派消息版本：{message.schema_version}")
    return message


__all__ = [
    "DISPATCH_SCHEMA_VERSION",
    "DispatchMessage",
    "decode_dispatch",
    "encode_dispatch",
    "new_dispatch",
]
