"""Writer 业务实现:不感知 langgraph 的纯函数。

装配与 loop 控制留在 agent.py,工具回执留在 tools.py;本模块所需全部经参数
注入(state/config/emit/render_incomplete),模型可见事件经 emit 回调发射。
"""

from collections.abc import Callable, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from deepresearcher.agents.writer.state import (
    PreparedEvidence,
    ValidatedDraft,
    WriterLoopContext,
    evidence_index_card,
)
from deepresearcher.config import AgentConfig
from deepresearcher.errors import AgentError
from deepresearcher.evidence.models import Evidence
from deepresearcher.observability.events import AgentEmit
from deepresearcher.observability.events.names import EventName
from deepresearcher.prompts import get_runtime_environment, render_data_section
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
from deepresearcher.vocab import SUPPORT_RANK

# 模型违反协议直接输出正文时，判定其可接收为草稿的下限：短于该长度或没有 cite 标记的
# 收尾文本按闲聊/致歉处理，不视为报告草稿。
INLINE_DRAFT_MIN_CHARS = 300


class WriterError(AgentError):
    """研究报告无法生成可审阅、可追溯的段落草稿。"""

    code = "writer_error"


class WriterGenerationError(WriterError):
    """LLM 或结构化输出层无法生成报告草稿，不能由改稿流程安全修复。"""

    code = "writer_generation"


def state_update_for_quick_answer(state: ResearchState) -> dict[str, object]:
    """即时回答直通交付：不检索、不引用，正文只保留答案本身。"""
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


def last_submitted_markdown(messages: Sequence[BaseMessage]) -> str:
    """取最后一次 CompleteReport 提交的正文,用于耗尽兜底时保留底稿。"""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            for call in message.tool_calls or []:
                if call["name"] == "CompleteReport":
                    return str(call.get("args", {}).get("markdown", ""))
    return ""


def recover_inline_draft(
    loop_context: WriterLoopContext,
    messages: Sequence[BaseMessage],
    *,
    emit: AgentEmit,
) -> None:
    """接收跳过 CompleteReport、把报告直接写成收尾正文的草稿。

    只有通过与 CompleteReport 完全相同的本地引用校验才算有效提交；
    校验失败时至少把正文保留进 last_markdown，不再整篇丢弃。
    """
    for message in reversed(list(messages)):
        if not isinstance(message, AIMessage) or message.tool_calls:
            continue
        text = str(message.text or "")
        if len(text) < INLINE_DRAFT_MIN_CHARS or "[[cite:" not in text.lower():
            continue
        loop_context.last_markdown = text
        if len(text) > loop_context.max_markdown_chars:
            loop_context.last_error = (
                f"模型直接输出的正文超过上限 {loop_context.max_markdown_chars} 字符，不予接收。"
            )
            emit(EventName.WRITER_INLINE_DRAFT_REJECTED, {"error": loop_context.last_error})
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
            emit(EventName.WRITER_INLINE_DRAFT_REJECTED, {"error": str(exc), "markdown": text})
            return
        cited = extract_cite_ids(text)
        selected = [
            item
            for item in dict.fromkeys(sorted(cited))
            if item in loop_context.read_evidence_ids
        ]
        if not selected:
            loop_context.last_error = "模型直接输出的正文未引用任何已读取 Evidence。"
            emit(EventName.WRITER_INLINE_DRAFT_REJECTED, {"error": loop_context.last_error})
            return
        loop_context.validated_draft = ValidatedDraft(body, bindings, citations, selected)
        emit(
            EventName.WRITER_INLINE_DRAFT_RECOVERED,
            {
                "selected_evidence_ids": selected,
                "markdown": text,
            },
        )
        return


def meets_minimum_support(config: AgentConfig, support: str) -> bool:
    """按 vocab 的 support 阶梯判定材料是否达到 Writer 最低可信等级。"""
    minimum_rank = SUPPORT_RANK.get(config.writer_minimum_support)
    return minimum_rank is not None and SUPPORT_RANK.get(support, 0) >= minimum_rank


def evidence_catalogue(
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


def prepare_evidence(
    config: AgentConfig,
    evidences: list[Evidence],
    report_brief: ReportBrief,
) -> PreparedEvidence:
    """过滤不满足可信等级的材料，并建立稳定的 Evidence ID 索引。"""
    usable = [item for item in evidences if meets_minimum_support(config, item.support)]
    by_id = {
        str(item.evidence_id or f"来源{index}"): item for index, item in enumerate(usable, 1)
    }
    return PreparedEvidence(
        by_id=by_id,
        catalogue=evidence_catalogue(by_id, report_brief),
    )


def require_writer_directive(state: ResearchState) -> WriterDirective:
    """WriterDirective 是写作的唯一入口事实源;缺席即不可开始。"""
    directive = state.get("writer_directive")
    if directive is None:
        raise WriterGenerationError("Supervisor 未提供 WriterDirective，不能开始报告写作。")
    return (
        directive
        if isinstance(directive, WriterDirective)
        else WriterDirective.model_validate(directive)
    )


def build_generation_messages(
    directive: WriterDirective,
    evidence_catalogue_text: str,
    *,
    writer_feedback_chars: int,
) -> list[BaseMessage]:
    """构造写作回合的首条 Human 消息:任务书、约束、上一稿与 Evidence 目录。"""
    revision_feedback = ""
    if directive.revision_instructions:
        feedback = "；".join(directive.revision_instructions)[:writer_feedback_chars]
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
                + evidence_catalogue_text
                + "\n</evidence_catalogue>"
            )
        )
    ]


def state_update_for_insufficient_evidence(
    state: ResearchState,
    evidences: list[Evidence],
    *,
    config: AgentConfig,
    render_incomplete: Callable[[ResearchState], str],
) -> dict[str, object]:
    """无可用材料时以降级报告收束,如实说明 support 不足。"""
    support_message = ""
    if evidences:
        support_message = (
            f"当前 Evidence 均低于 Writer 的最低支持等级 `{config.writer_minimum_support}`，"
            "不会用于事实性报告。"
        )
    report = render_incomplete(state)
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


def state_update_for_exhausted_result(
    state: ResearchState,
    *,
    last_markdown: str,
    error: str,
    config: AgentConfig,
    emit: AgentEmit,
) -> dict[str, object]:
    """回合耗尽:发 WRITER_EXHAUSTED 事件并以 rejected_draft 留底稿。"""
    emit(
        EventName.WRITER_EXHAUSTED,
        {
            "attempts": config.writer_max_turns,
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
        rejected_draft=last_markdown,
        # 新草稿开启下一次审阅，但不得清零跨草稿的累计次数；
        # 否则 Supervisor 的审阅恢复上限永远无法生效。
        review=ReviewProgress(status="pending", attempts=review_attempts),
    ).state_update()


def state_update_for_ready_result(
    state: ResearchState,
    *,
    draft: ValidatedDraft,
    evidence_count: int,
    emit: AgentEmit,
) -> dict[str, object]:
    """校验通过的草稿:发 WRITER_DRAFT_READY 并交审阅。"""
    source_count = len({item.url for item in draft.citations if item.url})
    emit(
        EventName.WRITER_DRAFT_READY,
        {
            "selected_evidence_ids": draft.selected_evidence_ids,
            "markdown": draft.body,
            "citation_ids": [item.id for item in draft.citations],
        },
    )
    review_attempts = section(state, "review", ReviewProgress).attempts
    return WriterResult(
        report_title=draft.title,
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
