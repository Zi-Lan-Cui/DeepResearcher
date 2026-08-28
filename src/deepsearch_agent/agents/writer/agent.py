"""Report Writer：将 Supervisor 提供的任务书与 Evidence 写成可审阅草稿。

Writer 只产出 evidence_id 键的草稿（report_draft）、段落绑定与引用元数据；
编号渲染与参考来源表由审阅通过后的终检渲染层完成，Writer 不渲染最终报告。
"""

from collections.abc import Callable, Sequence
from typing import Any, cast

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.errors import GraphRecursionError

from deepsearch_agent.agents.middleware import (
    AGENT_RECURSION_LIMIT,
    MiddlewareProfile,
    SubmissionGuard,
    build_agent_middleware,
)
from deepsearch_agent.agents.writer.state import (
    PreparedEvidence,
    ValidatedDraft,
    WriterRuntimeContext,
)
from deepsearch_agent.agents.writer.tools import build_writer_tools
from deepsearch_agent.config import AgentConfig
from deepsearch_agent.errors import WriterGenerationError
from deepsearch_agent.evidence.models import Evidence
from deepsearch_agent.llm import LLMConfigurationError, LLMInvoker
from deepsearch_agent.observability.events import bounded_content, emit_agent_event
from deepsearch_agent.observability.events.sink import JsonlSink
from deepsearch_agent.observability.logger import get_logger
from deepsearch_agent.reporting.validation import extract_cite_ids, validate_and_bind
from deepsearch_agent.schemas import (
    ResearchProgress,
    ReviewProgress,
    RunLifecycle,
    WriterDirective,
    WriterProgress,
    WriterResult,
)
from deepsearch_agent.state import ResearchState, section

_SUPPORT_RANK = {"insufficient": 0, "partial": 1, "direct": 2}
# 模型违反协议直接输出正文时的救回下限：短于该长度或没有 cite 标记的
# 收尾文本按闲聊/致歉处理，不视为报告草稿。
_INLINE_DRAFT_MIN_CHARS = 300


_WRITER_SYSTEM_PROMPT = "\n".join(
    [
    "【角色与边界】",
    "你是深度研究报告作者。Supervisor 已完成充分性判断并提供报告任务书；"
    "你必须按其 covered_topics、required_points 与 caveats 写作，"
    "而不重新决定是否研究或要求补材料。",
    "【引用协议（必须遵守）】",
    "凡是来自 Evidence 的可验证事实、数字、观点归属、具体案例，都必须在对应句子或段落末尾写"
    "[[cite:evidence_id1,evidence_id2]]。cite 内只能使用输入中已有的 evidence_id，最多三个，以逗号分隔。",
    "先从 Evidence 目录按研究方向、claim 与问题相关性选择要读取的 Evidence，调用 ReadEvidence 获取完整内容；"
    "只有读取返回的 evidence_id 才能引用。材料足够后必须调用 CompleteReport，"
    "将真正支撑正文的 evidence_id 填入 selected_evidence_ids。",
    "CompleteReport 是唯一的结束信号；在调用 CompleteReport 之前，不要直接输出报告文本。"
    "每轮只能调用一个工具。若上一次工具调用收到错误反馈，必须根据反馈修正后再次调用工具。",
    "不要手写 [来源N]，不要把 cite 放入代码围栏、行内代码或 URL。",
    "正确示例：Evidence 给出‘e1：某作品于 2020 年发行’，应写："
    "该作品于 2020 年发行。[[cite:e1]]",
    "不要写：该作品于 2020 年发行。[来源1]；"
    "不要写：该作品于 2020 年发行。[[cite:未知来源]]。",
    "【写作要求】",
    "只使用给定的已验证 Evidence 写成围绕用户问题展开的正式中文研究报告，"
    "不能按来源逐条罗列资料，也不能写成只有结论句的资料摘要。",
    "你不能凭目录中的 claim 补写 quote 未提供的细节；需要完整依据时先调用 ReadEvidence。",
    "如果研究状态为 incomplete 或报告生成模式为 partial，必须在报告中明确说明覆盖范围、未解决问题和证据限制；"
    "不要把局部证据写成完整综述，不要用模型内部知识填补缺口。",
    "除非 Evidence 明显不足，输出 3 到 4 个有信息量的小节；每节 2 到 4 个完整段落。"
    "报告应先直接回应问题，再形成清楚的论证链：界定讨论对象、解释证据与问题的关系、"
    "比较不同情况或观点，并说明适用范围与限制。",
    "每个段落通常用 2 到 5 句完成一个完整论点，而不是把单条 Evidence 改写成一句话。"
    "目标是约 1,000 到 1,800 个中文字；不得用重复、空泛修辞或外部知识凑篇幅。"
    "省略与问题无关的 Evidence。",
    "直接输出 Markdown 正文：可按论证需要使用 ##/### 标题、列表、引用块和表格；"
    "只有确实存在可比较的多个对象或维度时才使用表格，表格之后必须有解释，不能为排版而造表。",
    "不要输出 # 一级标题、‘研究问题’、‘参考来源’或文末参考文献，这些由本地程序统一生成。",
    "纯粹的衔接、范围说明或方法限制可以不加 cite，但绝不能新增外部事实。"
    "不得编造、不得扩大事实的时间、范围或条件；不能把推荐、题材特征自动写成价值证明。",
    "段落必须可独立阅读、自然衔接，并解释引用 Evidence 在本段论证中的作用。",
    "【输出前最后检查】",
    "1. selected_evidence_ids 非空且只含 evidence_id；"
    "2. 每个 cite 标记使用 selected_evidence_ids 中的 evidence_id；"
    "3. 每个 cite 必须是完整的 [[cite:id]]；"
    "4. 不输出‘参考来源’小节；"
    "5. 不用无引用的外部知识补全事实。",
    ]
)


class ReportWriter:
    """生成报告草稿，并只重试 Writer 自身可修复的引用协议错误。

    Supervisor 决定研究是否结束并提供 ``report_brief``；Writer 只负责基于
    给定 Evidence 组织文章。引用协议校验由 reporting 层提供，编号渲染
    发生在审阅通过后的终检渲染节点。
    """

    def __init__(
        self,
        llm: LLMInvoker,
        config: AgentConfig,
        *,
        render_incomplete: Callable[[ResearchState], str],
        event_sink: JsonlSink | None = None,
        artifact_max_text_chars: int = 1_000,
        context_window_tokens: int = 32_768,
    ):
        if llm is None:
            raise LLMConfigurationError("ReportWriter 需要已装配的 LLMInvoker。")
        self.llm = llm
        self.config = config
        self._render_incomplete = render_incomplete
        self._event_sink = event_sink
        self._artifact_max_text_chars = artifact_max_text_chars
        self._logger = get_logger("deepsearch_agent.agents.writer")
        self._agent_loop = create_agent(
            model=cast(Any, self.llm),
            tools=build_writer_tools(),
            system_prompt=_WRITER_SYSTEM_PROMPT,
            context_schema=WriterRuntimeContext,
            middleware=cast(Any, build_agent_middleware(MiddlewareProfile(
                agent_name="Writer",
                model=getattr(self.llm, "chat_model", None),
                max_turns=self.config.writer_max_turns,
                context_window_tokens=context_window_tokens,
                # 提交前输出纯文本不算结束：踢回重试，耗尽后由 _recover_inline_draft 兜底。
                submission_guard=SubmissionGuard(
                    nudge_message=(
                        "你还没有调用 CompleteReport，本任务尚未结束；直接输出正文不算提交。"
                        "请立即调用 CompleteReport，把完整 Markdown 正文作为参数提交，"
                        "并在 selected_evidence_ids 填入真正支撑正文的已读 evidence_id。"
                    ),
                    submitted_probe=lambda ctx: getattr(ctx, "validated_draft", None) is not None,
                ),
                emit=self._emit,
            ))),
            name="writer",
        )

    async def run(self, state: ResearchState) -> dict[str, object]:
        """按固定路径完成模式分流、草稿生成、校验和最终渲染。"""
        if state.get("answer_mode") == "quick_answer":
            return self._render_quick_answer(state)

        # 同 run 内 State channel 已是模型；跨进程 checkpoint 恢复时可能是 dict。
        evidences = [
            item if isinstance(item, Evidence) else Evidence.model_validate(item)
            for item in state.get("evidences", [])
        ]
        directive = self._require_writer_directive(state)
        if directive.evidence_ids is not None:
            allowed_ids = set(directive.evidence_ids)
            evidences = [item for item in evidences if item.evidence_id in allowed_ids]
        prepared = self._prepare_evidence(evidences)
        if not prepared.by_id:
            return self._render_insufficient_evidence(state, evidences)

        runtime = self._writer_runtime_context(prepared.by_id)
        messages = self._build_generation_messages(
            state=state,
            directive=directive,
            evidence_catalogue=prepared.catalogue,
        )
        try:
            result = await self._agent_loop.ainvoke(
                cast(Any, {"messages": messages}),
                context=runtime,
                config={"recursion_limit": AGENT_RECURSION_LIMIT},
            )
        except GraphRecursionError:
            result = {"messages": []}
            runtime.last_error = runtime.last_error or "Writer 回合预算耗尽，仍未提交有效报告。"
        if runtime.validated_draft is None:
            self._recover_inline_draft(runtime, result.get("messages", []))
        if runtime.validated_draft is None:
            return self._render_exhausted_result(
                state,
                last_markdown=runtime.last_markdown
                or self._last_submitted_markdown(result.get("messages", [])),
                error=runtime.last_error or "Writer 未提交有效报告。",
            )
        self._emit(
            "writer_draft_validated",
            {
                "read_evidence_ids": sorted(runtime.read_evidence_ids),
                "selected_evidence_ids": runtime.validated_draft.selected_evidence_ids,
                "normalized_markdown": runtime.validated_draft.body,
            },
        )
        return self._render_ready_result(
            state=state,
            draft=runtime.validated_draft,
            evidence_count=len(prepared.by_id),
        )

    def _writer_runtime_context(self, evidence_by_id: dict[str, Evidence]) -> WriterRuntimeContext:
        return WriterRuntimeContext(
            evidence_by_id=evidence_by_id,
            emit=self._emit,
            read_evidence_ids=set(),
            read_batch_size=self.config.writer_read_batch_size,
            max_selected_evidence=self.config.writer_max_selected_evidence,
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
        self, runtime: WriterRuntimeContext, messages: Sequence[BaseMessage]
    ) -> None:
        """救回跳过 CompleteReport、把报告直接写成收尾正文的草稿。

        只有通过与 CompleteReport 完全相同的本地引用校验才算有效提交；
        校验失败时至少把正文保留进 last_markdown，不再整篇丢弃。
        """
        for message in reversed(list(messages)):
            if not isinstance(message, AIMessage) or message.tool_calls:
                continue
            text = str(message.text or "")
            if len(text) < _INLINE_DRAFT_MIN_CHARS or "[[cite:" not in text.lower():
                continue
            runtime.last_markdown = text
            if len(text) > runtime.max_markdown_chars:
                runtime.last_error = (
                    f"模型直接输出的正文超过上限 {runtime.max_markdown_chars} 字符，未予救回。"
                )
                self._emit("writer_inline_draft_rejected", {"error": runtime.last_error})
                return
            try:
                body, bindings, citations = validate_and_bind(
                    text,
                    {
                        item: runtime.evidence_by_id[item]
                        for item in runtime.read_evidence_ids
                        if item in runtime.evidence_by_id
                    },
                )
            except ValueError as exc:
                runtime.last_error = f"模型直接输出正文而未提交，本地引用校验亦未通过：{exc}"
                self._emit("writer_inline_draft_rejected", {"error": str(exc), "markdown": text})
                return
            cited = extract_cite_ids(text)
            selected = [
                item
                for item in dict.fromkeys(sorted(cited))
                if item in runtime.read_evidence_ids
            ]
            if not selected:
                runtime.last_error = "模型直接输出的正文未引用任何已读取 Evidence。"
                self._emit("writer_inline_draft_rejected", {"error": runtime.last_error})
                return
            runtime.validated_draft = ValidatedDraft(body, bindings, citations, selected)
            self._emit(
                "writer_inline_draft_recovered",
                {
                    "selected_evidence_ids": selected,
                    "markdown": text,
                },
            )
            return

    def _render_quick_answer(self, state: ResearchState) -> dict[str, object]:
        lines = [
            "# 即时回答",
            "",
            "## 问题",
            state.get("clarified_query", state.get("query", "")),
            "",
            "> 以下内容基于模型已有知识生成，未进行联网检索或来源核验。",
            "> 它可能不完整或过时，不应作为可引用的研究结论。",
            "",
            "## 回答",
            state.get("draft_answer", "当前无法生成即时回答。"),
            "",
            "如需可验证来源、比较分析或完整清单，请进行深度研究。",
        ]
        return WriterResult(
            report="\n".join(lines),
            citations=[],
            answer_mode="quick_answer",
            current_round=0,
            evidence_count=0,
            source_count=0,
            run=RunLifecycle(phase="rendering"),
            writer=WriterProgress(status="completed", attempts=1),
        ).state_update()

    def _prepare_evidence(self, evidences: list[Evidence]) -> PreparedEvidence:
        """过滤不满足可信等级的材料，并建立稳定的 Evidence ID 索引。"""
        usable = [item for item in evidences if self._meets_minimum_support(item.support)]
        by_id = {
            str(item.evidence_id or f"来源{index}"): item for index, item in enumerate(usable, 1)
        }
        return PreparedEvidence(by_id=by_id, catalogue=self._evidence_catalogue(by_id))

    def _render_insufficient_evidence(
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
        research = section(state, "research", ResearchProgress)
        if support_message:
            report += f"\n\n- {support_message}"
        return WriterResult(
            report=report,
            citations=[],
            answer_mode="research_incomplete",
            run=RunLifecycle(phase="rendering"),
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
        revision_note = ""
        if directive.revision_instructions:
            feedback = "；".join(directive.revision_instructions)[: self.config.writer_feedback_chars]
            revision_note = (
                "【上一稿审阅意见】以下意见已经由 Supervisor 判定为应通过改写处理；"
                f"必须修正其中的 fatal 问题：{feedback}"
            )
        user_prompt = (
            f"原问题（必须直接回答，不能改成另一道题）：{directive.query}\n"
            f"Supervisor 的报告任务书：{directive.report_brief}\n"
            f"研究状态：{directive.research_status}；报告生成模式：{directive.generation_mode}\n"
            f"已知研究缺口（必须诚实保留，不得自行补全）：{directive.known_gaps}\n"
            f"上一稿（如有，必须在其基础上修订）：\n{directive.previous_draft}\n"
            f"可选 Evidence 目录：\n{evidence_catalogue}"
        )
        return [HumanMessage(content="\n".join(part for part in [revision_note, user_prompt] if part))]

    def _render_exhausted_result(
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
        return WriterResult(
            run=RunLifecycle(phase="rendering"),
            writer=WriterProgress(
                status="exhausted",
                attempts=section(state, "writer", WriterProgress).attempts + 1,
                failure_kind="citation_protocol",
                feedback=error,
                selected_evidence_ids=[],
            ),
            writer_draft=last_markdown,
            review=ReviewProgress(status="pending"),
        ).state_update()

    def _render_ready_result(
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
                "draft_markdown": draft.body,
                "citation_ids": [item.id for item in draft.citations],
            },
        )
        return WriterResult(
            report_draft=draft.body,
            citations=draft.citations,
            paragraph_bindings=draft.paragraph_bindings,
            answer_mode="deep_research",
            current_round=section(state, "research", ResearchProgress).current_round,
            evidence_count=evidence_count,
            source_count=source_count,
            run=RunLifecycle(phase="reviewing"),
            writer=WriterProgress(
                status="completed",
                attempts=section(state, "writer", WriterProgress).attempts + 1,
                selected_evidence_ids=draft.selected_evidence_ids,
            ),
            review=ReviewProgress(status="pending"),
        ).state_update()

    @staticmethod
    def _evidence_catalogue(evidence_by_id: dict[str, Evidence]) -> str:
        """按研究方向暴露紧凑索引；完整 quote 仅留给本地引用审计。"""
        groups: dict[str, list[str]] = {}
        for evidence_id, item in evidence_by_id.items():
            direction = item.research_direction or "未分类研究方向"
            entry = (
                f"- evidence_id={evidence_id} | "
                f"来源={item.source_title or '未命名来源'} | support={item.support} | "
                f"retrieval={item.retrieval_method} | "
                f"confidence={item.confidence}\n  claim：{item.claim}"
            )
            groups.setdefault(direction, []).append(entry)
        return "\n\n".join(
            f"[研究方向] {direction}\n" + "\n".join(entries)
            for direction, entries in groups.items()
        )

    def _meets_minimum_support(self, support: str) -> bool:
        minimum_rank = _SUPPORT_RANK.get(self.config.writer_minimum_support)
        return minimum_rank is not None and _SUPPORT_RANK.get(support, 0) >= minimum_rank

    @staticmethod
    def _citation_retry_note(error: str) -> str:
        return (
            f"\n上一稿 Markdown 引用校验失败：{error}。"
            "如果提示 Evidence 尚未读取，必须先调用 ReadEvidence 获取它；"
            "然后重新生成完整 Markdown 正文。不要复用旧的引用写法，"
            "必须使用 [[cite:evidence_id]] 句末标记。"
        )

    def _emit(self, event_type: str, payload: dict[str, object]) -> None:
        """记录 Writer 生命周期元数据，并为长文本保留受限预览。"""
        emit_agent_event(
            self._event_sink,
            self._logger,
            event_type,
            bounded_content(payload, max_text_chars=self._artifact_max_text_chars),
            component="writer",
            node_fallback="writer",
        )
