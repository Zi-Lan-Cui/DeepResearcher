"""服务层测试共享件（风格对齐 tests/fakes.py：显式构造，不用 conftest）。"""

from __future__ import annotations

import json
from types import SimpleNamespace

from deepresearcher.config import (
    AgentConfig,
    AppConfig,
    LLMConfig,
    ObservabilityConfig,
    SearchConfig,
    Settings,
)
from deepresearcher.service.settings import ServiceConfig


def service_settings(tmp_path) -> Settings:
    return Settings(
        llm=LLMConfig(),
        agent=AgentConfig(),
        search=SearchConfig(tavily_api_key="test"),
        app=AppConfig(),
        observability=ObservabilityConfig(log_dir=tmp_path),
    )


def service_config(tmp_path, **overrides) -> ServiceConfig:
    base = dict(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'service.db'}",
        jwt_secret="s" * 40,
        service_log_dir=tmp_path,
        jsonl_events=False,
        max_concurrent_runs_per_user=2,
    )
    base.update(overrides)
    return ServiceConfig(**base)


class FakeGraph:
    """假研究图：先发若干真实形态的引擎事件（run_id 注入）、可选等门控、返回结果或抛错。"""

    def __init__(self, *, result=None, error=None, gate=None, emit=()):
        self.result = result if result is not None else completed_result()
        self.error = error
        self.gate = gate
        self.emit = list(emit)
        self.ainvoke_inputs: list[dict] = []
        self._sink = None

    async def _run(self, state):
        self.ainvoke_inputs.append(dict(state))
        for event in self.emit:
            if self._sink is not None:
                self._sink.write({**event, "run_id": state["run_id"]})
        if self.gate is not None:
            await self.gate.wait()
        if self.error is not None:
            raise self.error
        return self.result

    async def ainvoke(self, state, **_kwargs):
        return await self._run(state)

    async def astream(self, state, **_kwargs):
        yield ((), "values", await self._run(state))


def completed_result() -> dict:
    return {
        "run": SimpleNamespace(phase="completed", terminal_reason="report_rendered", error=None),
        "answer_mode": "deep_research",
        "report": "# 研究报告\n结论。",
        "citations": [{"id": "e1", "url": "https://a", "title": "A", "quote": "q", "claim": "c"}],
        "evidence_count": 3,
        "source_count": 2,
    }


def parse_sse(text: str) -> list[tuple[str, dict]]:
    """把 SSE 响应文本解析成 (event, data) 序列；忽略 : ping 注释帧。"""
    frames: list[tuple[str, dict]] = []
    for block in text.split("\n\n"):
        block = block.strip("\n")
        if not block or block.startswith(":"):
            continue
        event, data = "", "{}"
        for line in block.splitlines():
            if line.startswith("id: "):
                continue
            if line.startswith("event: "):
                event = line[len("event: ") :]
            elif line.startswith("data: "):
                data = line[len("data: ") :]
        frames.append((event, json.loads(data)))
    return frames
