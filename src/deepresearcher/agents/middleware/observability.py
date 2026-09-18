"""Agent 模型回合、工具调用与终止的统一可观测性中间件。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from time import monotonic
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, ToolMessage

from deepresearcher.context.execution import AgentExecutionScope
from deepresearcher.observability.logger import get_logger
from deepresearcher.observability.usage_runtime import enforce_usage_budget

_LIMIT_MESSAGE_MARKER = "Model call limits exceeded"
LIMIT_MESSAGE_MARKER = _LIMIT_MESSAGE_MARKER
_PREVIEW_CHARS = 800
_ERROR_PREVIEW_CHARS = 400
_TOOL_ARGUMENT_PREVIEW_CHARS = 2_000
_SENSITIVE_TOOL_ARGUMENTS = {"AddEvidence"}
_SENSITIVE_TOOL_RESULTS = {
    "ListSearchResults",
    "ReadSources",
    "GrepDocument",
    "ReadDocument",
    "AddEvidence",
}


class AgentObservabilityMiddleware(AgentMiddleware):
    """记录 Agent 回合、工具调用边界和终止原因。"""

    def __init__(
        self,
        agent_name: str,
        run_limit: int,
        emit: Callable[[str, dict[str, object]], None] | None = None,
    ):
        super().__init__()
        self.agent_name = agent_name
        self.run_limit = run_limit
        self._emit = emit
        self._logger = get_logger("deepresearcher.agents.middleware.observability")

    @staticmethod
    def _call_count(state: Any) -> int:
        if not isinstance(state, dict):
            return 0
        return int(state.get("run_model_call_count", 0) or 0)

    def _event_context(self, runtime: Any) -> dict[str, object]:
        context = getattr(runtime, "context", None)
        scope = getattr(context, "scope", None)
        fields = scope.event_fields() if isinstance(scope, AgentExecutionScope) else {}
        fields["agent"] = self.agent_name
        return fields

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        """在每个 Agent 模型回合前检查 run/user/platform 预算。"""
        await enforce_usage_budget()
        return await handler(request)

    async def aafter_model(self, state: Any, runtime: Any) -> None:
        messages = state.get("messages", []) if isinstance(state, dict) else []
        last = messages[-1] if messages else None
        if not isinstance(last, AIMessage):
            return
        turn = self._call_count(state)
        content = last.text or ""
        self._log_event(
            f"{self.agent_name.lower()}_model_turn",
            {
                **self._event_context(runtime),
                "turn": turn,
                "model_call_count": turn,
                "run_limit": self.run_limit,
                "tool_names": [str(call.get("name", "")) for call in last.tool_calls or []],
                "content_chars": len(content),
                "content_preview": content[:_PREVIEW_CHARS],
            },
        )

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        """记录工具调用边界，不改变返回值、异常或取消语义。"""
        tool_call = request.tool_call
        tool_name = str(tool_call.get("name", ""))
        base = {
            **self._event_context(request.runtime),
            "tool_name": tool_name,
            "tool_call_id": str(tool_call.get("id", "")),
            "turn": self._call_count(request.state),
            "argument_keys": sorted(str(key) for key in (tool_call.get("args") or {}))
            if isinstance(tool_call.get("args"), dict)
            else [],
            "arguments_preview": (
                ""
                if tool_name in _SENSITIVE_TOOL_ARGUMENTS
                else self._preview(tool_call.get("args"))
            ),
            "arguments_redacted": tool_name in _SENSITIVE_TOOL_ARGUMENTS,
        }
        started_at = monotonic()
        self._log_event(f"{self.agent_name.lower()}_tool_started", base)
        try:
            result = await handler(request)
        except asyncio.CancelledError:
            self._log_event(
                f"{self.agent_name.lower()}_tool_failed",
                {
                    **base,
                    "duration_ms": self._duration_ms(started_at),
                    "error_type": "CancelledError",
                    "cancelled": True,
                },
            )
            raise
        except Exception as exc:
            self._log_event(
                f"{self.agent_name.lower()}_tool_failed",
                {
                    **base,
                    "duration_ms": self._duration_ms(started_at),
                    "error_type": type(exc).__name__,
                    "error_preview": str(exc)[:_ERROR_PREVIEW_CHARS],
                    "cancelled": False,
                },
            )
            raise
        self._log_event(
            f"{self.agent_name.lower()}_tool_completed",
            {
                **base,
                "duration_ms": self._duration_ms(started_at),
                **self._result_summary(result, redact=tool_name in _SENSITIVE_TOOL_RESULTS),
            },
        )
        return result

    async def aafter_agent(self, state: Any, runtime: Any) -> None:
        messages = state.get("messages", []) if isinstance(state, dict) else []
        last = messages[-1] if messages else None
        reason = "final_response"
        if isinstance(last, AIMessage):
            content = last.text or ""
            if _LIMIT_MESSAGE_MARKER in content:
                reason = "model_call_limit_exceeded"
            elif last.tool_calls:
                reason = "ended_on_tool_call_turn"
        self._log_event(
            f"{self.agent_name.lower()}_agent_finished",
            {
                **self._event_context(runtime),
                "turns": self._call_count(state),
                "run_limit": self.run_limit,
                "stop_reason": reason,
            },
        )

    @staticmethod
    def _duration_ms(started_at: float) -> int:
        return max(0, round((monotonic() - started_at) * 1_000))

    @classmethod
    def _result_summary(cls, result: Any, *, redact: bool = False) -> dict[str, object]:
        summary: dict[str, object] = {"result_type": type(result).__name__}
        if isinstance(result, ToolMessage):
            content = result.content
            summary["content_chars"] = len(content) if isinstance(content, str) else 0
            summary["content_redacted"] = redact
            if not redact:
                summary["content_preview"] = cls._preview(content)
                # grep/search 的 total_matches/has_more/candidate_count 等标量排在
                # 大数组(matches/candidates)之后,常被 content_preview(800 字符)截断
                # 而看不到 → 无法从日志审计取证漏斗与翻页。把标量 hoist 到事件顶层:
                # 只搬标量/短串,不改工具返回值、不进模型上下文,纯可观测性增强。
                metrics = cls._result_metrics(content)
                if metrics:
                    summary["result_metrics"] = metrics
        return summary

    @staticmethod
    def _result_metrics(content: object) -> dict[str, object]:
        if not isinstance(content, str):
            return {}
        body = content.split("\n", 1)[-1] if content.startswith("【系统工具执行结果") else content
        try:
            parsed = json.loads(body)
        except (TypeError, ValueError):
            return {}
        if not isinstance(parsed, dict):
            return {}
        return {
            key: value
            for key, value in parsed.items()
            if isinstance(value, (bool, int, float)) or (isinstance(value, str) and len(value) <= 48)
        }

    @staticmethod
    def _preview(value: Any) -> str:
        if value is None:
            return ""
        try:
            rendered = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            rendered = str(value)
        return rendered[:_TOOL_ARGUMENT_PREVIEW_CHARS]

    def _log_event(self, event_type: str, payload: dict[str, object]) -> None:
        try:
            if self._emit is not None:
                self._emit(event_type, payload)
                return
            self._logger.info("%s payload=%s", event_type, payload)
        except Exception:
            # 可观测性是旁路：sink 异常不得阻断、重试或改写工具执行。
            self._logger.exception("%s emission failed", event_type)
