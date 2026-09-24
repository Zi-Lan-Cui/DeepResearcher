"""Report Writer：将 Supervisor 提供的任务书与 Evidence 写成可审阅草稿。

Writer 只产出 evidence_id 键的草稿（report_draft）、段落绑定与引用元数据；
编号渲染与参考来源表由审阅通过后的终检渲染层完成，Writer 不渲染最终报告。
"""

from collections.abc import Callable, Sequence
from typing import Any, cast

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.errors import GraphRecursionError

from deepresearcher.agents.middleware import (
    AGENT_RECURSION_LIMIT,
    MiddlewareProfile,
    SubmissionGuard,
    build_agent_middleware,
)
from deepresearcher.agents.writer.state import (
    PreparedEvidence,
    ValidatedDraft,
    WriterLoopContext,
    evidence_index_card,
)
from deepresearcher.agents.writer.tools import build_writer_tools
from deepresearcher.config import DEFAULT_CONTEXT_WINDOW_TOKENS, AgentConfig
from deepresearcher.errors import AgentError
from deepresearcher.evidence.models import Evidence
from deepresearcher.llm import LLMConfigurationError
from deepresearcher.observability.events import bounded_content, emit_agent_event
from deepresearcher.observability.events.sink import JsonlSink
from deepresearcher.observability.execution import AgentExecutionScope
from deepresearcher.observability.logging_config import get_logger
from deepresearcher.prompts import (
    get_runtime_environment,
    language_directive,
    load_prompt,
    render_data_section,
)
from deepresearcher.reporting.validation import extract_cite_ids, validate_and_bind
from deepresearcher.schemas import (
    ReportBrief,
    ReviewProgress,
    RunStatus,
    SupervisorProgress,
    WriterDirective,
    WriterProgress,
    WriterResult,
)
from deepresearcher.state import ResearchState, section
from deepresearcher.vocab import SUPPORT_RANK as _SUPPORT_RANK

# 模型违反协议直接输出正文时，判定其可接收为草稿的下限：短于该长度或没有 cite 标记的
# 收尾文本按闲聊/致歉处理，不视为报告草稿。
_INLINE_DRAFT_MIN_CHARS = 300


_WRITER_SYSTEM_PROMPT = load_prompt("writer")


class WriterError(AgentError):
    """研究报告无法生成可审阅、可追溯的段落草稿。"""

    code = "writer_error"


class WriterGenerationError(WriterError):
    """LLM 或结构化输出层无法生成报告草稿，不能由改稿流程安全修复。"""

    code = "writer_generation"


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
        if llm is None:
            raise LLMConfigurationError("ReportWriter 需要已装配的模型。")
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
            middleware=cast(
                Any,
                build_agent_middleware(
                    MiddlewareProfile(
                        agent_name="Writer",
                        event_slug="writer",
                        model=self.llm,
                        max_turns=self.config.writer_max_turns,
                        context_window_tokens=context_window_tokens,
                        # 提交前输出纯文本不算结束：退回重试，耗尽后由 _recover_inline_draft 兜底。
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
                        # 校验通过的草稿落定后下一跳静默出环:收场不再依赖
                        # 模型多跑一轮自由文本,也关掉"提交后仍可再调工具"的窗口。
                        exit_probe=lambda ctx, _state: (
                            getattr(ctx, "validated_draft", None) is not None
                        ),
                        emit=self._emit,
                    )
                ),
            ),
            name="writer",
        )

    async def run(self, state: ResearchState) -> dict[str, object]:
        """按固定路径完成模式分流、草稿生成、校验与状态收束;编号渲染不在本层。"""
        if state.get("answer_mode") == "quick_answer":
            return self._state_update_for_quick_answer(state)

        # 同 run 内 State channel 已是模型；跨进程 checkpoint 恢复时可能是 dict。
        evidences = [
            item if isinstance(item, Evidence) else Evidence.model_validate(item)
            for item in state.get("evidences", [])
        ]
        directive = self._require_writer_directive(state)
        if directive.evidence_ids is not None:
            allowed_ids = set(directive.evidence_ids)
            evidences = [item for item in evidences if item.evidence_id in allowed_ids]
        prepared = self._prepare_evidence(evidences, directive.report_brief)
        if not prepared.by_id:
            return self._state_update_for_insufficient_evidence(state, evidences)

        loop_context = self._build_loop_context(
            prepared.by_id,
            run_id=str(state.get("run_id") or ""),
        )
        messages = self._build_generation_messages(
            state=state,
            directive=directive,
            evidence_catalogue=prepared.catalogue,
        )
        try:
            result = await self._agent_loop.ainvoke(
                cast(Any, {"messages": messages}),
                context=loop_context,
                config={"recursion_limit": AGENT_RECURSION_LIMIT},
            )
        except GraphRecursionError:
            result = {"messages": []}
            loop_context.last_error = (
                loop_context.last_error or "Writer 回合预算耗尽，仍未提交有效报告。"
            )
        if loop_context.validated_draft is None:
            self._recover_inline_draft(loop_context, result.get("messages", []))
        if loop_context.validated_draft is None:
            return self._state_update_for_exhausted_result(
                state,
                last_markdown=loop_context.last_markdown
                or self._last_submitted_markdown(result.get("messages", [])),
                error=loop_context.last_error or "Writer 未提交有效报告。",
            )
        self._emit(
            "writer_draft_validated",
            {
                "read_evidence_ids": sorted(loop_context.read_evidence_ids),
                "selected_evidence_ids": loop_context.validated_draft.selected_evidence_ids,
                "normalized_markdown": loop_context.validated_draft.body,
            },
        )
        return self._state_update_for_ready_result(
            state=state,
            draft=loop_context.validated_draft,
            evidence_count=len(prepared.by_id),
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
            artifact_max_text_chars=self._artifact_max_text_chars,
        )

    @staticmethod
    def _last_submitted_markdown(messages: Sequence[BaseMessage]) -> str:
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                for call in message.tool_calls or []:
                    if call["name"] == "CompleteReport":
                        return str(call.get("args", {}).get("markdown", ""))
        return ""

    def _recover_inline_draft(
        self, loop_context: WriterLoopContext, messages: Sequence[BaseMessage]
    ) -> None:
        """接收跳过 CompleteReport、把报告直接写成收尾正文的草稿。

        只有通过与 CompleteReport 完全相同的本地引用校验才算有效提交；
        校验失败时至少把正文保留进 last_markdown，不再整篇丢弃。
        """
        for message in reversed(list(messages)):
            if not isinstance(message, AIMessage) or message.tool_calls:
                continue
            text = str(message.text or "")
            if len(text) < _INLINE_DRAFT_MIN_CHARS or "[[cite:" not in text.lower():
                continue
            loop_context.last_markdown = text
            if len(text) > loop_context.max_markdown_chars:
                loop_context.last_error = (
                    f"模型直接输出的正文超过上限 {loop_context.max_markdown_chars} 字符，不予接收。"
                )
                self._emit("writer_inline_draft_rejected", {"error": loop_context.last_error})
                return
            try:
                body, bindings, citations = validate_and_bind(
                    text,
                    {
                        item: loop_context.evidence_by_id[item]
                        for item in loop_context.read_evidence_ids
                        if item in loop_context.evidence_by_id
                    },
                )
            except ValueError as exc:
                loop_context.last_error = f"模型直接输出正文而未提交，本地引用校验亦未通过：{exc}"
                self._emit("writer_inline_draft_rejected", {"error": str(exc), "markdown": text})
                return
            cited = extract_cite_ids(text)
            selected = [
                item
                for item in dict.fromkeys(sorted(cited))
                if item in loop_context.read_evidence_ids
            ]
            if not selected:
                loop_context.last_error = "模型直接输出的正文未引用任何已读取 Evidence。"
                self._emit("writer_inline_draft_rejected", {"error": loop_context.last_error})
                return
            loop_context.validated_draft = ValidatedDraft(body, bindings, citations, selected)
            self._emit(
                "writer_inline_draft_recovered",
                {
                    "selected_evidence_ids": selected,
                    "markdown": text,
                },
            )
            return

    def _state_update_for_quick_answer(self, state: ResearchState) -> dict[str, object]:
        return WriterResult(
            # 页面徽标已经表达“即时回答 / 未联网检索”；正文只保留答案，
            # 不重复问题、模式说明或行动号召。
            report=str(state.get("draft_answer") or "当前无法生成即时回答。"),
            citations=[],
            answer_mode="quick_answer",
            current_round=0,
            evidence_count=0,
            source_count=0,
            run=RunStatus(phase="rendering"),
            writer=WriterProgress(status="completed", attempts=1),
        ).state_update()

    def _prepare_evidence(
        self,
        evidences: list[Evidence],
        report_brief: ReportBrief,
    ) -> PreparedEvidence:
        """过滤不满足可信等级的材料，并建立稳定的 Evidence ID 索引。"""
        usable = [item for item in evidences if self._meets_minimum_support(item.support)]
        by_id = {
            str(item.evidence_id or f"来源{index}"): item for index, item in enumerate(usable, 1)
        }
        return PreparedEvidence(
            by_id=by_id,
            catalogue=self._evidence_catalogue(by_id, report_brief),
        )

    def _state_update_for_insufficient_evidence(
        self,
        state: ResearchState,
        evidences: list[Evidence],
    ) -> dict[str, object]:
        support_message = ""
        if evidences:
            support_message = (
                f"当前 Evidence 均低于 Writer 的最低支持等级 `{self.config.writer_minimum_support}`，"
                "不会用于事实性报告。"
            )
        report = self._render_incomplete(state)
        research = section(state, "supervisor", SupervisorProgress)
        if support_message:
            report += f"\n\n- {support_message}"
        return WriterResult(
            report=report,
            citations=[],
            answer_mode="research_incomplete",
            run=RunStatus(phase="rendering"),
            writer=WriterProgress(
                status="failed",
                attempts=section(state, "writer", WriterProgress).attempts + 1,
                failure_kind="insufficient_evidence",
                feedback=support_message or "没有可用于写作的 Evidence。",
            ),
            current_round=research.current_round,
            evidence_count=0,
            source_count=0,
        ).state_update()

    @staticmethod
    def _require_writer_directive(state: ResearchState) -> WriterDirective:
        directive = state.get("writer_directive")
        if directive is None:
            raise WriterGenerationError("Supervisor 未提供 WriterDirective，不能开始报告写作。")
        return (
            directive
            if isinstance(directive, WriterDirective)
            else WriterDirective.model_validate(directive)
        )

    def _build_generation_messages(
        self,
        *,
        state: ResearchState,
        directive: WriterDirective,
        evidence_catalogue: str,
    ) -> list[BaseMessage]:
        revision_feedback = ""
        if directive.revision_instructions:
            feedback = "；".join(directive.revision_instructions)[
                : self.config.writer_feedback_chars
            ]
            revision_feedback = (
                "以下意见已由 Supervisor 判定为应通过改写处理；"
                f"必须修正其中的 fatal 问题：{feedback}"
            )
        report_context = {
            "原问题（必须直接回答）": directive.query,
            "Supervisor 的报告任务书": directive.report_brief,
            "研究状态": directive.research_status,
            "报告生成模式": directive.generation_mode,
            "已知研究缺口（不得自行补全）": directive.known_gaps,
            "上一稿（如有，必须在其基础上修订）": directive.previous_draft,
            "上一稿审阅意见": revision_feedback,
        }
        return [
            HumanMessage(
                content=(
                    render_data_section("运行时环境", get_runtime_environment().payload())
                    + "\n\n---\n\n"
                    + render_data_section("报告任务与约束", report_context)
                    + "\n\n---\n\n## 可选 Evidence 目录\n\n"
                    + "<evidence_catalogue>\n"
                    + evidence_catalogue
                    + "\n</evidence_catalogue>"
                )
            )
        ]

    def _state_update_for_exhausted_result(
        self,
        state: ResearchState,
        *,
        last_markdown: str,
        error: str,
    ) -> dict[str, object]:
        self._emit(
            "writer_exhausted",
            {
                "attempts": self.config.writer_max_turns,
                "failure_kind": "citation_protocol",
                "error": error,
                "markdown": last_markdown,
            },
        )
        review_attempts = section(state, "review", ReviewProgress).attempts
        return WriterResult(
            run=RunStatus(phase="rendering"),
            writer=WriterProgress(
                status="exhausted",
                attempts=section(state, "writer", WriterProgress).attempts + 1,
                failure_kind="citation_protocol",
                feedback=error,
                selected_evidence_ids=[],
            ),
            writer_draft=last_markdown,
            # 新草稿开启下一次审阅，但不得清零跨草稿的累计次数；
            # 否则 Supervisor 的审阅恢复上限永远无法生效。
            review=ReviewProgress(status="pending", attempts=review_attempts),
        ).state_update()

    def _state_update_for_ready_result(
        self,
        *,
        state: ResearchState,
        draft: ValidatedDraft,
        evidence_count: int,
    ) -> dict[str, object]:
        source_count = len({item.url for item in draft.citations if item.url})
        self._emit(
            "writer_draft_ready",
            {
                "selected_evidence_ids": draft.selected_evidence_ids,
                "markdown": draft.body,
                "citation_ids": [item.id for item in draft.citations],
            },
        )
        review_attempts = section(state, "review", ReviewProgress).attempts
        return WriterResult(
            report_draft=draft.body,
            citations=draft.citations,
            paragraph_bindings=draft.paragraph_bindings,
            answer_mode="deep_research",
            current_round=section(state, "supervisor", SupervisorProgress).current_round,
            evidence_count=evidence_count,
            source_count=source_count,
            run=RunStatus(phase="reviewing"),
            writer=WriterProgress(
                status="completed",
                attempts=section(state, "writer", WriterProgress).attempts + 1,
                selected_evidence_ids=draft.selected_evidence_ids,
            ),
            review=ReviewProgress(status="pending", attempts=review_attempts),
        ).state_update()

    @staticmethod
    def _evidence_catalogue(
        evidence_by_id: dict[str, Evidence],
        report_brief: ReportBrief | None = None,
    ) -> str:
        """暴露主题到 Evidence 的归属与去重索引；完整 quote 按需读取。"""
        topics = list(report_brief.covered_topics) if report_brief is not None else []
        has_assignments = any(topic.evidence_ids for topic in topics)
        if has_assignments:
            topic_plan = "\n\n".join(
                "\n".join(
                    [
                        f"[写作主题] {topic.topic}",
                        f"作用：{topic.role}",
                        f"Supervisor 综合：{topic.reason}",
                        "建议 Evidence："
                        + ", ".join(
                            evidence_id
                            for evidence_id in topic.evidence_ids
                            if evidence_id in evidence_by_id
                        ),
                    ]
                )
                for topic in topics
            )
            ordered_ids = list(
                dict.fromkeys(
                    evidence_id
                    for topic in topics
                    for evidence_id in topic.evidence_ids
                    if evidence_id in evidence_by_id
                )
            )
        else:
            # 旧 checkpoint 没有 CoveredTopic.evidence_ids，退化为全局索引。
            topic_plan = ""
            ordered_ids = list(evidence_by_id)

        entries: list[str] = []
        for evidence_id in ordered_ids:
            card = evidence_index_card(evidence_by_id[evidence_id])
            published_metadata = (
                f" | published_at={card['published_at']}(搜索元信息)"
                if "published_at" in card
                else ""
            )
            entries.append(
                f"- evidence_id={evidence_id} | "
                f"来源={card['source_title'] or '未命名来源'} | "
                f"domain={card['source_domain'] or 'unknown'} | "
                f"source_type={card['source_type']} | support={card['support']}"
                f"{published_metadata}\n  claim：{card['claim']}"
            )
        evidence_index = "[Evidence 去重索引]\n" + "\n".join(entries)
        return "\n\n".join(part for part in (topic_plan, evidence_index) if part)

    def _meets_minimum_support(self, support: str) -> bool:
        minimum_rank = _SUPPORT_RANK.get(self.config.writer_minimum_support)
        return minimum_rank is not None and _SUPPORT_RANK.get(support, 0) >= minimum_rank

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
