"""结构化输出的重试边界：只接管解析与校验失败，其余错误原样向上抛。"""

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel

from deepresearcher.llm import ainvoke_structured


class _Payload(BaseModel):
    value: int


class _Runnable:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[list[Any]] = []

    async def ainvoke(self, messages):
        self.calls.append(list(messages))
        item = self.outcomes.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _LLM:
    def __init__(self, outcomes: list[Any]) -> None:
        self.runnable = _Runnable(outcomes)

    def with_structured_output(self, schema, method=None):
        assert method == "json_mode"
        return self.runnable


@pytest.fixture(autouse=True)
def _attempt_budget(monkeypatch):
    """预算固定为 2（含首次），用例不依赖外部环境里的 LLM_STRUCTURED_MAX_ATTEMPTS。"""
    monkeypatch.setattr(
        "deepresearcher.llm.get_settings",
        lambda: SimpleNamespace(llm=SimpleNamespace(structured_max_attempts=2)),
    )


def test_repairs_once_after_parse_failure():
    llm = _LLM([OutputParserException("bad json"), _Payload(value=1)])

    result = asyncio.run(ainvoke_structured(llm, _Payload, []))

    assert result == _Payload(value=1)
    assert len(llm.runnable.calls) == 2
    # 第二次请求在原始消息之后追加了一条回灌的修复提示。
    assert len(llm.runnable.calls[1]) == len(llm.runnable.calls[0]) + 1
    assert "结构校验" in llm.runnable.calls[1][-1].content


def test_raises_the_last_error_when_budget_is_exhausted():
    llm = _LLM([OutputParserException("first"), OutputParserException("second")])

    with pytest.raises(OutputParserException):
        asyncio.run(ainvoke_structured(llm, _Payload, []))

    assert len(llm.runnable.calls) == 2


def test_non_structured_error_is_not_retried():
    llm = _LLM([RuntimeError("传输层故障")])

    with pytest.raises(RuntimeError):
        asyncio.run(ainvoke_structured(llm, _Payload, []))

    assert len(llm.runnable.calls) == 1
