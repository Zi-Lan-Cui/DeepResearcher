"""Agent 模型回合、工具调用与终止的统一可观测性中间件。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from time import monotonic
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, ToolMessage

from deepresearcher.observability.execution import AgentExecutionScope
from deepresearcher.observability.logging_config import get_logger
from deepresearcher.schemas.limits import EVENT_CONTENT_PREVIEW_CHARS
from deepresearcher.schemas.tool_args import TOOL_RECEIPT_PREFIX
from deepresearcher.timing import elapsed_ms
from deepresearcher.usage_runtime import enforce_usage_budget

LIMIT_MESSAGE_MARKER = "Model call limits exceeded"
_ERROR_PREVIEW_CHARS = 400
_TOOL_ARGUMENT_PREVIEW_CHARS = 2_000
_SENSITIVE_TOOL_ARGUMENTS = {"AddEvidence", "CompleteReport"}
_SENSITIVE_TOOL_RESULTS = {
    "ListSearchResults",
    "ReadSources",
    "GrepDocument",
    "ReadDocument",
    "AddEvidence",
    "ReadEvidence",
}


class AgentObservabilityMiddleware(AgentMiddleware):
    """记录 Agent 回合、工具调用边界和终止原因。"""

    def __init__(
        self,
        agent_name: str,
        run_limit: int,
        emit: Callable[[str, dict[str, object]], None] | None = None,
        event_slug: str | None = None,
    ):
        super().__init__()
        self.agent_name = agent_name
        # 事件名前缀走显式 slug：显示名是驼峰，lower() 之后仍得不到 snake_case 形式。
        self._slug = event_slug or agent_name.lower()
        self.run_limit = run_limit
        self._emit = emit
        self._logger = get_logger("deepresearcher.agents.middleware.observability")
        # awrap_model_call 探到的回合耗时按 AIMessage id 挂账，aafter_model 取走。
        # 跳回/驳回丢弃的消息条目靠上限淘汰；单次请求的账在 run_usage，两处不互推。
        self._turn_durations: dict[str, int] = {}

    @staticmethod
    def _call_count(state: Any) -> int:
        if not isinstance(state, dict):
            return 0
        return int(state.get("run_model_call_count", 0) or 0)

    def _event_context(self, runtime: Any) -> dict[str, object]:
        # context 契约按 getattr 读取，删改 LoopContext 字段前先 grep 本文件。
        # scope: AgentExecutionScope | None，所有 agent 的 LoopContext 必须携带；
        # 模型回合与工具事件的 run_id/task_id/operation_id 归因依赖它，缺席时不崩溃但静默失去归属。
        context = getattr(runtime, "context", None)
        scope: AgentExecutionScope | None = getattr(context, "scope", None)
        fields = scope.event_fields() if isinstance(scope, AgentExecutionScope) else {}
        fields["agent"] = self.agent_name
        return fields

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        """在每个 Agent 模型回合前检查预算，并记录本回合的墙上耗时。

        计时包含位于本中间件下层的重试等待，计时范围是"这个回合花了多久";
        单次 HTTP 尝试的账在 run_usage，两处不互推。
        """
        await enforce_usage_budget()
        started = monotonic()
        response = await handler(request)
        duration_ms = elapsed_ms(started)
        for message in self._response_messages(response):
            if isinstance(message, AIMessage) and message.id:
                self._turn_durations[str(message.id)] = duration_ms
                if len(self._turn_durations) > 64:
                    self._turn_durations.pop(next(iter(self._turn_durations)))
        return response

    @staticmethod
    def _response_messages(response: Any) -> list[Any]:
        """兼容 handler 返回 AIMessage 本体或含 result/messages 列表的响应对象。"""
        if isinstance(response, AIMessage):
            return [response]
        for attribute in ("result", "messages"):
            value = getattr(response, attribute, None)
            if isinstance(value, list):
                return value
        return []

    async def aafter_model(self, state: Any, runtime: Any) -> None:
        messages = state.get("messages", []) if isinstance(state, dict) else []
        last = messages[-1] if messages else None
        if not isinstance(last, AIMessage):
            return
        turn = self._call_count(state)
        content = last.text or ""
        duration_ms = self._turn_durations.pop(str(last.id), None)
        payload: dict[str, object] = {
            **self._event_context(runtime),
            "turn": turn,
            "run_limit": self.run_limit,
            "tool_names": [str(call.get("name", "")) for call in last.tool_calls or []],
            "content_chars": len(content),
            "content_preview": content[:EVENT_CONTENT_PREVIEW_CHARS],
        }
        if duration_ms is not None:
            payload["duration_ms"] = duration_ms
        self._log_event(f"{self._slug}_model_turn", payload)

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
        self._log_event(f"{self._slug}_tool_started", base)
        try:
            result = await handler(request)
        except asyncio.CancelledError:
            self._log_event(
                f"{self._slug}_tool_failed",
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
                f"{self._slug}_tool_failed",
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
            f"{self._slug}_tool_completed",
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
            if LIMIT_MESSAGE_MARKER in content:
                reason = "model_call_limit_exceeded"
            elif last.tool_calls:
                reason = "ended_on_tool_call_turn"
        self._log_event(
            f"{self._slug}_agent_finished",
            {
                **self._event_context(runtime),
                "turns": self._call_count(state),
                "run_limit": self.run_limit,
                "stop_reason": reason,
            },
        )

    @staticmethod
    def _duration_ms(started_at: float) -> int:
        return elapsed_ms(started_at)

    @classmethod
    def _result_summary(cls, result: Any, *, redact: bool = False) -> dict[str, object]:
        summary: dict[str, object] = {"result_type": type(result).__name__}
        if isinstance(result, ToolMessage):
            content = result.content
            summary["content_chars"] = len(content) if isinstance(content, str) else 0
            summary["content_redacted"] = redact
            if not redact:
                summary["content_preview"] = cls._preview(content)
            # 计数排在 matches/candidates 大数组之后会被 content_preview 截断，取证漏斗与翻页
            # 因此无从审计，所以提到事件顶层。脱敏时计数照提，但只允许 int/float/bool。
            metrics = cls._result_metrics(content, numeric_only=redact)
            if metrics:
                summary["result_metrics"] = metrics
        return summary

    @staticmethod
    def _result_metrics(content: object, *, numeric_only: bool = False) -> dict[str, object]:
        if not isinstance(content, str):
            return {}
        body = content.split("\n", 1)[-1] if content.startswith(TOOL_RECEIPT_PREFIX) else content
        try:
            parsed = json.loads(body)
        except (TypeError, ValueError):
            return {}
        if not isinstance(parsed, dict):
            return {}
        return {
            key: value
            for key, value in parsed.items()
            if isinstance(value, (bool, int, float))
            or (not numeric_only and isinstance(value, str) and len(value) <= 48)
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
        # 事件旁路的异常容错在 emit_agent_event 一层，此处不再各自设防。
        if self._emit is not None:
            self._emit(event_type, payload)
            return
        self._logger.info("%s payload=%s", event_type, payload)
