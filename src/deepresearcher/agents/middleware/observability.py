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
from deepresearcher.observability.logger import get_logger
from deepresearcher.observability.usage_runtime import enforce_usage_budget
from deepresearcher.schemas.tool_args import TOOL_RECEIPT_PREFIX

LIMIT_MESSAGE_MARKER = "Model call limits exceeded"
_PREVIEW_CHARS = 800
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
        # 事件名前缀走显式 slug,不再从显示名 lower() 派生("ResearchAgent"→驼峰假蛇形)。
        self._slug = event_slug or agent_name.lower()
        self.run_limit = run_limit
        self._emit = emit
        self._logger = get_logger("deepresearcher.agents.middleware.observability")

    @staticmethod
    def _call_count(state: Any) -> int:
        if not isinstance(state, dict):
            return 0
        return int(state.get("run_model_call_count", 0) or 0)

    def _event_context(self, runtime: Any) -> dict[str, object]:
        # context 契约(鸭子读,删改 LoopContext 字段前先 grep 本文件):
        #   scope: AgentExecutionScope | None —— 所有 agent 的 LoopContext 必须携带,
        #   模型回合/工具事件的 run_id/task_id/operation_id 归因全靠它;缺席是静默降级。
        context = getattr(runtime, "context", None)
        scope: AgentExecutionScope | None = getattr(context, "scope", None)
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
            f"{self._slug}_model_turn",
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
            # total_matches/has_more/candidate_count 等计数排在 matches/candidates 大数组
            # 之后,常被 content_preview 截断而看不到 → 无法审计取证漏斗与翻页。把标量 hoist
            # 到事件顶层。**即使正文被 redact(读取类工具含来源原文引用)也照提取计数**——
            # 数字本身不敏感,不该被正文脱敏一起挡掉;此时只放行 int/float/bool,不放行字符串。
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
        # 旁路保护在 emit_agent_event 一层，此处不再各自设防。
        if self._emit is not None:
            self._emit(event_type, payload)
            return
        self._logger.info("%s payload=%s", event_type, payload)
