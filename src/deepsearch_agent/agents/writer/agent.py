"""Report Writer：将 Supervisor 提供的任务书与 Evidence 写成可审阅草稿。

Writer 只产出 evidence_id 键的草稿（report_draft）、段落绑定与引用元数据；
编号渲染与参考来源表由审阅通过后的终检渲染层完成，Writer 不渲染最终报告。
"""

import json
from collections.abc import Callable, Mapping, Sequence
from typing import ClassVar

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field, ValidationError

from deepsearch_agent.agents.writer.state import (
    GenerationResult,
    PreparedEvidence,
    ValidatedDraft,
)
from deepsearch_agent.config import AgentConfig
from deepsearch_agent.context import ContextPolicy
from deepsearch_agent.evidence.models import Evidence
from deepsearch_agent.llm import LLMConfigurationError, LLMInvoker
from deepsearch_agent.observability.events import make_audit_event
from deepsearch_agent.observability.events.sink import JsonlSink
from deepsearch_agent.observability.logger import get_logger
from deepsearch_agent.observability.tracing.context import current_context
from deepsearch_agent.orchestration.errors import WriterGenerationError
from deepsearch_agent.reporting.validation import (
    DraftProtocolError,
    extract_cite_ids,
    validate_and_bind,
)
from deepsearch_agent.schemas import (
    MarkdownReportDraft,
    ResearchProgress,
    ReviewProgress,
    RunLifecycle,
    WriterDirective,
    WriterProgress,
    WriterResult,
)
from deepsearch_agent.state import ResearchState, section

_SUPPORT_RANK = {"insufficient": 0, "partial": 1, "direct": 2}


class ReadEvidence(BaseModel):
    """Writer 请求查看目录中 Evidence 的完整内容。"""

    allow_parallel: ClassVar[bool] = False

    evidence_ids: list[str] = Field(min_length=1, max_length=5)
    reason: str = Field(min_length=1, description="说明这些证据与当前报告段落的关系。")


class CompleteReport(BaseModel):
    """Writer 确认已读取足够证据并提交报告草稿。"""

    allow_parallel: ClassVar[bool] = False

    selected_evidence_ids: list[str] = Field(min_length=1)
    markdown: str = Field(min_length=1)


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
        context_policy: ContextPolicy | None = None,
    ):
        if llm is None:
            raise LLMConfigurationError("ReportWriter 需要已装配的 LLMInvoker。")
        self.llm = llm
        self.config = config
        self._render_incomplete = render_incomplete
        self._event_sink = event_sink
        self.context_policy = context_policy or ContextPolicy()
        self._artifact_max_text_chars = artifact_max_text_chars
        self._logger = get_logger("deepsearch_agent.agents.writer")
        self._decision_runnable = llm.bind_tools(
            [ReadEvidence, CompleteReport],
            tool_choice="any",
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

        generation = await self._generate_agent_validated_draft(
            state=state,
            directive=directive,
            evidence_catalogue=prepared.catalogue,
            evidence_by_id=prepared.by_id,
        )
        if generation.draft is None:
            return self._render_exhausted_result(state, generation)
        return self._render_ready_result(
            state=state,
            draft=generation.draft,
            evidence_count=len(prepared.by_id),
        )

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

    async def _generate_agent_validated_draft(
        self,
        *,
        state: ResearchState,
        directive: WriterDirective,
        evidence_catalogue: str,
        evidence_by_id: dict[str, Evidence],
    ) -> GenerationResult:
        """让 Writer 先按需读取 Evidence，再提交一次完整草稿。"""
        last_markdown = ""
        last_error = ""
        messages: list[BaseMessage] = self._build_generation_messages(
            state=state,
            directive=directive,
            evidence_catalogue=evidence_catalogue,
        )
        read_ids: set[str] = set()
        read_evidence: dict[str, Evidence] = {}
        for turn in range(1, self.config.writer_max_turns + 1):
            prepared_messages = await self.context_policy.aprepare(messages, agent="writer")
            response = await self._decision_runnable.ainvoke(prepared_messages)
            if not isinstance(response, AIMessage):
                raise WriterGenerationError("Writer 工具模型没有返回有效的 AIMessage。")
            calls = response.tool_calls or []
            messages.append(response)
            call = self._parse_single_tool_call(messages, calls, turn=turn)
            if call is None:
                continue
            if call["name"] == "CompleteReport":
                try:
                    completed = CompleteReport.model_validate(call.get("args", {}))
                except ValidationError as exc:
                    last_error = f"CompleteReport 参数无效：{exc}"
                    self._reject_call(
                        messages,
                        call,
                        f"{last_error}\n{self._citation_retry_note(last_error)}",
                    )
                    continue
                if len(completed.selected_evidence_ids) > self.config.writer_max_selected_evidence:
                    last_error = (
                        "CompleteReport 选择的 Evidence 数量超过配置上限："
                        f"{self.config.writer_max_selected_evidence}。"
                    )
                    self._reject_call(messages, call, last_error)
                    continue
                if len(completed.markdown) > self.config.writer_max_markdown_chars:
                    last_error = (
                        "CompleteReport Markdown 超过配置上限："
                        f"{self.config.writer_max_markdown_chars} 字符。"
                    )
                    self._reject_call(messages, call, last_error)
                    continue
                draft = MarkdownReportDraft(
                    markdown=completed.markdown,
                    selected_evidence_ids=completed.selected_evidence_ids,
                )
                last_markdown = draft.markdown
                try:
                    validated = self._validate_draft(
                        draft,
                        read_evidence,
                        available_evidence=evidence_by_id,
                    )
                except ValueError as exc:
                    last_error = str(exc)
                    self._emit(
                        "writer_citation_validation_failed",
                        {"turn": turn, "error": last_error, "markdown": last_markdown},
                    )
                    self._reject_call(
                        messages,
                        call,
                        f"{last_error}\n{self._citation_retry_note(last_error)}",
                    )
                    continue
                self._emit(
                    "writer_draft_validated",
                    {
                        "turn": turn,
                        "read_evidence_ids": sorted(read_ids),
                        "selected_evidence_ids": validated.selected_evidence_ids,
                        "normalized_markdown": validated.body,
                    },
                )
                return GenerationResult(validated, last_markdown, "")
            if call["name"] != "ReadEvidence":
                last_error = (
                    f"Writer 调用了未知工具：{call['name']}。只能调用 ReadEvidence 或 CompleteReport；"
                    "请重新选择一个合法工具。"
                )
                self._emit("writer_tool_protocol_error", {"turn": turn, "error": last_error})
                self._reject_call(messages, call, last_error)
                continue
            tool_result = self._execute_read_tool(
                call,
                evidence_by_id=evidence_by_id,
                read_ids=read_ids,
                read_evidence=read_evidence,
            )
            messages.append(
                ToolMessage(
                    content=json.dumps(tool_result, ensure_ascii=False),
                    name="ReadEvidence",
                    tool_call_id=call["id"],
                )
            )
        if not last_error:
            last_error = "Writer 工具轮次预算耗尽，仍未提交有效报告。"
        return GenerationResult(None, last_markdown, last_error)

    def _execute_read_tool(
        self,
        call: Mapping[str, object],
        *,
        evidence_by_id: dict[str, Evidence],
        read_ids: set[str],
        read_evidence: dict[str, Evidence],
    ) -> dict[str, object]:
        """解析并执行一次 ReadEvidence 调用，返回可写入 ToolMessage 的结果。"""
        try:
            request = ReadEvidence.model_validate(call.get("args", {}))
        except ValidationError as exc:
            return {"error": f"ReadEvidence 参数无效：{exc}"}
        return self._read_evidence(
            request.evidence_ids,
            evidence_by_id=evidence_by_id,
            read_ids=read_ids,
            read_evidence=read_evidence,
        )

    def _parse_single_tool_call(
        self,
        messages: list[BaseMessage],
        calls: Sequence[Mapping[str, object]],
        *,
        turn: int,
    ) -> Mapping[str, object] | None:
        """校验单轮工具协议；失败时把可修复错误写回对话并返回 None。"""
        if not calls:
            error = "Writer 没有调用工具。必须调用 ReadEvidence 或 CompleteReport，不能直接输出普通文本。"
            self._emit("writer_tool_protocol_error", {"turn": turn, "error": error})
            self._reject_call(messages, None, error)
            return None
        if len(calls) != 1:
            names = ", ".join(str(item.get("name", "unknown")) for item in calls)
            error = (
                f"Writer 一轮返回了多个工具调用（{names}）。每轮只能选择一个工具；"
                "请重新选择：需要材料时调用 ReadEvidence，材料足够时调用 CompleteReport。"
            )
            self._emit("writer_tool_protocol_error", {"turn": turn, "error": error})
            for item in calls:
                messages.append(
                    ToolMessage(
                        content=error,
                        name=str(item.get("name", "unknown")),
                        tool_call_id=str(item.get("id", "")),
                    )
                )
            return None
        return calls[0]

    @staticmethod
    def _reject_call(
        messages: list[BaseMessage],
        call: Mapping[str, object] | None,
        error: str,
    ) -> None:
        """把工具协议错误写回对话，避免重复发送完整错误内容。"""
        if call is not None:
            messages.append(
                ToolMessage(
                    content=error,
                    name=str(call.get("name", "unknown")),
                    tool_call_id=str(call.get("id", "")),
                )
            )
        else:
            messages.append(HumanMessage(content="请调用合法工具，并根据错误提示修正后重试。"))

    def _read_evidence(
        self,
        requested_ids: list[str],
        *,
        evidence_by_id: dict[str, Evidence],
        read_ids: set[str],
        read_evidence: dict[str, Evidence],
    ) -> dict[str, object]:
        ids = list(dict.fromkeys(requested_ids))[: self.config.writer_read_batch_size]
        unknown = [item for item in ids if item not in evidence_by_id]
        for evidence_id in ids:
            if evidence_id in evidence_by_id:
                read_ids.add(evidence_id)
                read_evidence[evidence_id] = evidence_by_id[evidence_id]
        self._emit(
            "writer_evidence_read",
            {"requested_ids": requested_ids, "read_ids": ids, "unknown_ids": unknown},
        )
        return {
            "evidence": [evidence_by_id[item].model_dump() for item in ids if item in evidence_by_id],
            "unknown_ids": unknown,
            "read_count": len(read_ids),
        }

    def _build_generation_messages(
        self,
        *,
        state: ResearchState,
        directive: WriterDirective,
        evidence_catalogue: str,
    ) -> list[BaseMessage]:
        system_prompt = self._system_prompt()
        if directive.revision_instructions:
            feedback = "；".join(directive.revision_instructions)[: self.config.writer_feedback_chars]
            system_prompt += (
                "\n【上一稿审阅意见】以下意见已经由 Supervisor 判定为应通过改写处理；"
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
        return [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

    def _validate_draft(
        self,
        draft: MarkdownReportDraft,
        read_evidence: dict[str, Evidence],
        *,
        available_evidence: dict[str, Evidence],
    ) -> ValidatedDraft:
        """校验引用，并区分不存在与尚未读取的 Evidence。"""
        cited_ids = extract_cite_ids(draft.markdown)
        unknown_ids = [item for item in cited_ids if item not in available_evidence]
        if unknown_ids:
            raise DraftProtocolError(
                "cite 使用了不存在的 Evidence："
                f"{', '.join(sorted(unknown_ids))}。"
            )
        unread_ids = [item for item in cited_ids if item not in read_evidence]
        if unread_ids:
            raise DraftProtocolError(
                "cite 使用了尚未读取的 Evidence："
                f"{', '.join(sorted(unread_ids))}。请先调用 ReadEvidence 读取这些 ID。"
            )
        selected_ids = self._selected_evidence_ids(
            draft.selected_evidence_ids, cited_ids, read_evidence
        )
        body, bindings, citations = validate_and_bind(draft.markdown, read_evidence)
        if not selected_ids:
            raise DraftProtocolError("Writer 没有声明或实际引用任何 Evidence。")
        return ValidatedDraft(
            body=body,
            paragraph_bindings=bindings,
            citations=citations,
            selected_evidence_ids=selected_ids,
        )

    @staticmethod
    def _selected_evidence_ids(
        declared_ids: list[str],
        cited_ids: set[str],
        evidence_by_id: dict[str, Evidence],
    ) -> list[str]:
        """正文有效 cite 是最终事实绑定；声明列表仅作为模型工作集提示。"""
        selected = list(
            dict.fromkeys(
                evidence_id.strip()
                for evidence_id in declared_ids
                if evidence_id.strip() in evidence_by_id
            )
        )
        selected.extend(
            evidence_id
            for evidence_id in cited_ids
            if evidence_id in evidence_by_id and evidence_id not in selected
        )
        return selected

    def _render_exhausted_result(
        self,
        state: ResearchState,
        generation: GenerationResult,
    ) -> dict[str, object]:
        error = generation.validation_error or "模型没有产出可解析的引用标记。"
        self._emit(
            "writer_exhausted",
            {
                "attempts": self.config.writer_max_turns,
                "failure_kind": "citation_protocol",
                "error": error,
                "markdown": generation.last_markdown,
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
            writer_draft=generation.last_markdown,
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
        context = current_context()
        content_keys = {"markdown", "normalized_markdown", "report"}
        event_payload = {key: value for key, value in payload.items() if key not in content_keys}
        for key in content_keys:
            if key in payload:
                value = str(payload[key])
                event_payload[f"{key}_chars"] = len(value)
                event_payload[f"{key}_preview"] = value[: self._artifact_max_text_chars]
        event = make_audit_event(
            event_type,
            trace_id=context.trace_id if context else None,
            span_id=context.span_id if context else None,
            run_id=context.run_id if context else None,
            session_id=context.session_id if context else None,
            node_id=context.node_id if context else "writer",
            component="writer",
            payload=event_payload,
        )
        if self._event_sink is not None:
            self._event_sink.write(event)
        self._logger.info("%s payload=%s", event_type, event_payload)

    def _system_prompt(self) -> str:
        return "\n".join(
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
