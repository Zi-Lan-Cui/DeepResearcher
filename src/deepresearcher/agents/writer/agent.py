"""Report Writer：将 Supervisor 提供的任务书与 Evidence 写成可审阅草稿。

Writer 只产出 evidence_id 键的草稿（report_draft）、段落绑定与引用元数据；
编号渲染与证据来源表由审阅通过后的终检渲染层完成，Writer 不渲染最终报告。
业务实现在 services.py，本模块只做装配、loop 控制与事件发射。
"""

from collections.abc import Callable
from typing import Any, cast

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.errors import GraphRecursionError

from deepresearcher.agents.middleware import (
    AGENT_RECURSION_LIMIT,
    LoopStateRebuildMiddleware,
    MiddlewareProfile,
    SubmissionGuard,
    build_agent_middleware,
)
from deepresearcher.agents.writer import services
from deepresearcher.agents.writer.state import (
    WriterAgentState,
    WriterLoopContext,
)
from deepresearcher.agents.writer.tools import build_writer_tools
from deepresearcher.config import DEFAULT_CONTEXT_WINDOW_TOKENS, AgentConfig
from deepresearcher.evidence.models import Evidence
from deepresearcher.observability.events import bounded_content, emit_agent_event
from deepresearcher.observability.events.names import EventName
from deepresearcher.observability.events.sink import JsonlSink
from deepresearcher.observability.execution import AgentExecutionScope
from deepresearcher.observability.logging_config import get_logger
from deepresearcher.prompts import language_directive, load_prompt
from deepresearcher.state import ResearchState

_WRITER_SYSTEM_PROMPT = load_prompt("writer")

# CompleteReport 写 replace/retain 语义通道,即草稿、错误、底稿,是串行工具:必须
# 单独一轮,与 ReadEvidence 等同批调用会被 SerialToolMiddleware 整批驳回、打回模型。
# 名单约束由 tests/test_writer.py 的静态守卫测试校验(方案 §5 分类守卫)。
WRITER_SERIAL_TOOLS: frozenset[str] = frozenset({"CompleteReport"})


class ReportWriter:
    """生成报告草稿，并只重试 Writer 自身可修复的引用协议错误。

    Supervisor 决定研究是否结束并交接 ``writer_directive``（含 report_brief）；
    Writer 只负责基于给定 Evidence 组织文章。引用协议校验由 reporting 层提供，
    编号渲染发生在审阅通过后的终检渲染节点。
    """

    def __init__(
        self,
        llm: BaseChatModel,
        config: AgentConfig,
        *,
        render_incomplete: Callable[[ResearchState], str],
        event_sink: JsonlSink | None = None,
        artifact_max_text_chars: int = 1_000,
        context_window_tokens: int = DEFAULT_CONTEXT_WINDOW_TOKENS,
    ):
        self.llm = llm
        self.config = config
        self._render_incomplete = render_incomplete
        self._event_sink = event_sink
        self._artifact_max_text_chars = artifact_max_text_chars
        self._logger = get_logger("deepresearcher.agents.writer")
        self._agent_loop = create_agent(
            model=self.llm,
            tools=build_writer_tools(
                turn_budget=config.writer_max_turns,
                read_batch=config.writer_read_batch_size,
            ),
            system_prompt=_WRITER_SYSTEM_PROMPT.replace("__LANG__", config.output_language)
            + "\n"
            + language_directive(config.output_language),
            context_schema=WriterLoopContext,
            state_schema=WriterAgentState,
            middleware=cast(
                Any,
                [
                    LoopStateRebuildMiddleware(),
                    *build_agent_middleware(
                        MiddlewareProfile(
                            agent_name="Writer",
                            event_slug="writer",
                            model=self.llm,
                            max_turns=self.config.writer_max_turns,
                            context_window_tokens=context_window_tokens,
                            serial_tools=set(WRITER_SERIAL_TOOLS),
                            # 提交前输出纯文本不算结束:退回重试,退回次数耗尽后由 recover_inline_draft 兜底。
                            submission_guard=SubmissionGuard(
                                nudge_message=(
                                    "你还没有调用 CompleteReport，本任务尚未结束；直接输出正文不算提交。"
                                    "请立即调用 CompleteReport，把完整 Markdown 正文作为参数提交，"
                                    "并在 selected_evidence_ids 填入真正支撑正文的已读 evidence_id。"
                                ),
                                submitted_probe=lambda ctx: (
                                    getattr(ctx, "validated_draft", None) is not None
                                ),
                                max_nudges=self.config.finalization_attempts,
                            ),
                            # 草稿落定后下一跳退出循环:不依赖模型多跑一轮,提交后模型也没有继续调用工具的机会。
                            exit_probe=lambda ctx, _state: (
                                getattr(ctx, "validated_draft", None) is not None
                            ),
                            emit=self._emit,
                        )
                    ),
                ],
            ),
            name="writer",
            # 业务状态经通道持久化,崩溃重放从最后一个已提交 superstep 续跑。
        )

    async def run(self, state: ResearchState) -> dict[str, object]:
        """按固定路径完成模式分流、草稿生成、校验与状态收束;编号渲染不在本层。"""
        if state.get("answer_mode") == "quick_answer":
            return services.state_update_for_quick_answer(state)

        # 同 run 内 State channel 已是模型；跨进程 checkpoint 恢复时可能是 dict。
        evidences = [
            item if isinstance(item, Evidence) else Evidence.model_validate(item)
            for item in state.get("evidences", [])
        ]
        directive = services.require_writer_directive(state)
        if directive.evidence_ids is not None:
            allowed_ids = set(directive.evidence_ids)
            evidences = [item for item in evidences if item.evidence_id in allowed_ids]
        prepared = services.prepare_evidence(self.config, evidences, directive.report_brief)
        if not prepared.by_id:
            return services.state_update_for_insufficient_evidence(
                state,
                evidences,
                config=self.config,
                render_incomplete=self._render_incomplete,
            )

        loop_context = self._build_loop_context(
            prepared.by_id,
            run_id=str(state.get("run_id") or ""),
        )
        messages = services.build_generation_messages(
            directive,
            prepared.catalogue,
            writer_feedback_chars=self.config.writer_feedback_chars,
        )
        try:
            result = await cast(Any, self._agent_loop).ainvoke(
                cast(Any, {"messages": messages, **loop_context.initial_channel_state()}),
                context=loop_context,
                config={"recursion_limit": AGENT_RECURSION_LIMIT},
            )
        except GraphRecursionError:
            result = {"messages": []}
            loop_context.last_error = (
                loop_context.last_error or "Writer 回合预算耗尽，仍未提交有效报告。"
            )
        else:
            # 从通道现值回填簿记:崩溃重放时上下文未经批重建,以通道为准可避免把已提交草稿读成 None。
            loop_context.rebuild_loop_state(result)
        if loop_context.validated_draft is None:
            services.recover_inline_draft(loop_context, result.get("messages", []), emit=self._emit)
        if loop_context.validated_draft is None:
            return services.state_update_for_exhausted_result(
                state,
                last_markdown=loop_context.last_markdown
                or services.last_submitted_markdown(result.get("messages", [])),
                error=loop_context.last_error or "Writer 未提交有效报告。",
                config=self.config,
                emit=self._emit,
            )
        self._emit(
            EventName.WRITER_DRAFT_VALIDATED,
            {
                "read_evidence_ids": sorted(loop_context.read_evidence_ids),
                "selected_evidence_ids": loop_context.validated_draft.selected_evidence_ids,
                "normalized_markdown": loop_context.validated_draft.body,
            },
        )
        return services.state_update_for_ready_result(
            state,
            draft=loop_context.validated_draft,
            evidence_count=len(prepared.by_id),
            emit=self._emit,
        )

    def _build_loop_context(
        self,
        evidence_by_id: dict[str, Evidence],
        *,
        run_id: str,
    ) -> WriterLoopContext:
        return WriterLoopContext(
            scope=AgentExecutionScope(run_id=run_id, agent_name="Writer"),
            evidence_by_id=evidence_by_id,
            emit=self._emit,
            read_evidence_ids=set(),
            read_batch_size=self.config.writer_read_batch_size,
            max_markdown_chars=self.config.writer_max_markdown_chars,
        )

    def _emit(
        self,
        event_type: str,
        payload: dict[str, object],
        *,
        component: str = "writer",
    ) -> None:
        """记录 Writer 生命周期元数据，并为长文本保留受限预览。"""
        emit_agent_event(
            self._event_sink,
            self._logger,
            event_type,
            bounded_content(payload, max_text_chars=self._artifact_max_text_chars),
            component=component,
            node_fallback="writer",
        )
