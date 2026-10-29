"""工具调用入参与工具结果契约。"""

from __future__ import annotations

import json
import re
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from deepresearcher.evidence.models import normalize_evidence_ids
from deepresearcher.schemas.limits import (
    CLARIFICATION_ASSUMPTION_LIMIT,
    CLARIFICATION_INTENT_MAX_CHARS,
    CLARIFICATION_OPTION_COUNT,
    CLARIFICATION_QUESTION_MAX_CHARS,
    EVIDENCE_REFERENCES_HARD_LIMIT,
    READ_EVIDENCE_REQUEST_WINDOW_IDS,
    REPORT_TITLE_MAX_CHARS,
    RUN_HEADLINE_MAX_CHARS,
    SEARCH_QUERIES_PER_CALL,
    SEARCH_RESULTS_PAGE_HARD_LIMIT,
    SEARCH_RESULTS_PREVIEW_COUNT,
    SOURCE_CANDIDATES_PER_READ,
    STRUCTURED_COLLECTION_HARD_LIMIT,
    STRUCTURED_TEXT_HARD_LIMIT_CHARS,
    TASK_TITLE_MAX_CHARS,
)
from deepresearcher.schemas.reporting import PlanItem, ResearchAspect
from deepresearcher.vocab import Support


class SearchSources(BaseModel):
    """Researcher 请求发现当前任务的候选来源。"""

    reason: str = Field(description="为什么这些检索式能补足当前任务的证据。")
    queries: list[str] = Field(
        min_length=1,
        max_length=SEARCH_QUERIES_PER_CALL,
        description=(
            "一到两条针对当前任务缺口的短检索式；写成自然关键词串，"
            "不要加英文引号包裹短语，也不要 AND/OR/NOT、+/-、括号等布尔语法。"
        ),
    )


class ReadSources(BaseModel):
    """Researcher 从候选目录中选择实际读取的来源。"""

    candidate_ids: list[str] = Field(
        min_length=1,
        max_length=SOURCE_CANDIDATES_PER_READ,
        description="要读取的候选来源 ID；只能使用 SearchSources 返回的 ID。",
    )
    reason: str = Field(description="说明这些来源与当前任务缺口的关系。")


class ListSearchResults(BaseModel):
    """分页查看已写入磁盘的搜索结果。"""

    search_id: str = Field(min_length=1, description="SearchSources 返回的搜索结果句柄。")
    offset: int = Field(default=0, ge=0, description="从第几条结果开始，从 0 计数。")
    limit: int = Field(
        default=SEARCH_RESULTS_PREVIEW_COUNT,
        ge=1,
        le=SEARCH_RESULTS_PAGE_HARD_LIMIT,
        description="本次返回的结果数。",
    )
    reason: str = Field(description="为什么需要继续查看该批搜索结果。")


class _DocumentIdSanitizer(BaseModel):
    """持有 document_id 入参的模型的共同底座：模型回填长 id 时偶发夹入空白与控制字符，先归一再比对。"""

    # check_fields=False：字段由各自子类声明，基类单独建类时不校验存在性。
    @field_validator("document_id", mode="before", check_fields=False)
    @classmethod
    def strip_injected_whitespace_and_control_chars(cls, value: object) -> object:
        if isinstance(value, str):
            return re.sub(r"[\s\x00-\x1f\x7f]", "", value)
        return value


class GrepWindowSegment(BaseModel):
    """归并命中流上的一段：跳过 offset 个命中窗口，返回至多 count 个。"""

    offset: int = Field(default=0, ge=0, description="从归并命中流的第 offset 个窗口起。")
    count: int = Field(
        default=12,
        ge=1,
        le=32,
        description="本段返回的命中窗口数；超出工具每段上限的部分被截断。",
    )


class GrepDocument(_DocumentIdSanitizer):
    """在已读取长文中按多个普通文本词定位相关行；命中按文档行号归并成一条流。

    多个关键词一次列进 patterns，不按词拆成多次调用。segments 可在同一次调用里取多个
    不相邻的段。offset 恒指归并流位置。超长行与窗口尾部会裁切并标注，被裁内容不在
    后续段里，按标注行号用 ReadDocument 补读。
    """

    document_id: str = Field(
        min_length=1,
        description="目标文档的 document_id，来自 ReadSources 登记返回；未知时先调 ReadWorkingSet 核对。",
    )
    patterns: list[str] = Field(
        min_length=1,
        max_length=8,
        description="普通文本关键词列表；单个词也写成一个元素的列表。",
    )
    segments: list[GrepWindowSegment] = Field(
        default_factory=lambda: [GrepWindowSegment()],
        min_length=1,
        max_length=4,
        description="要读的命中流窗口段；一段为常规，两段可在一轮内比较首尾。",
    )
    context_lines: int = Field(default=2, ge=0, le=10)
    reason: str = Field(description="关键词与当前证据缺口的关系。")


class DocumentLineRange(BaseModel):
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)


class DocumentReadTarget(_DocumentIdSanitizer):
    """多文档批量读取的单位：一个文档号与其行段集合。"""

    document_id: str = Field(
        min_length=1,
        description="目标文档的 document_id，来自 ReadSources 登记返回。",
    )
    ranges: list[DocumentLineRange] = Field(min_length=1, max_length=32)


class ReadDocument(BaseModel):
    """按行号跨多份已登记文档批量读取有限窗口；程序逐文档展开执行。"""

    documents: list[DocumentReadTarget] = Field(
        min_length=1,
        max_length=4,
        description="要读取的文档及其行段；读多份文档时列进同一次调用，不拆成多轮。",
    )
    reason: str = Field(description="为什么需要读取这些行段。")


class EvidenceSubmission(_DocumentIdSanitizer):
    """Researcher 从已经看到的文档原文中提出一条原子 Evidence：只交首尾片段锚，quote 由程序逐字回填。"""

    document_id: str = Field(
        min_length=1,
        description="必须是 ReadSources 登记返回的 document_id；不在其中时先重新读取来源。",
    )
    claim: str = Field(min_length=1, max_length=STRUCTURED_TEXT_HARD_LIMIT_CHARS)
    quote_start: str = Field(
        min_length=1,
        max_length=STRUCTURED_TEXT_HARD_LIMIT_CHARS,
        description="引用开头的几个词。必须能在该文档原文中唯一命中；命中多处时加长片段直到无歧义。",
    )
    quote_end: str = Field(
        default="",
        max_length=STRUCTURED_TEXT_HARD_LIMIT_CHARS,
        description="引用结尾的几个词，须出现在 quote_start 之后；省略时引用就是 quote_start 命中的片段本身。",
    )
    support: Support = "strict"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class AddEvidence(BaseModel):
    """批量提交 Evidence 候选；系统逐字验证后才会写入。"""

    evidences: list[EvidenceSubmission] = Field(min_length=1, max_length=32)
    reason: str = Field(description="这些 Evidence 如何回答当前任务。")


class ReadWorkingSet(BaseModel):
    """查看当前 Agent 工作集的轻量摘要（Researcher 侧）。"""

    reason: str = Field(default="", description="说明需要重新检查工作集的原因。")


class ReadWorkbench(BaseModel):
    """Supervisor 重读工作台全景的入参。"""

    reason: str = Field(default="", description="说明需要重新查看工作台的原因。")


class ReleaseEvidence(BaseModel):
    """从当前 Agent 活跃工作集释放 Evidence；不删除 Evidence 档案。"""

    evidence_ids: list[str] = Field(min_length=1)
    reason: str = Field(description="说明这些 Evidence 为什么应从当前工作集中释放。")

    @field_validator("evidence_ids", mode="before")
    @classmethod
    def normalize_evidence_id_lists(cls, value: object) -> object:
        return normalize_evidence_ids(value)


class RestoreEvidence(BaseModel):
    """将 Evidence 档案中的候选重新放回当前活跃工作集。"""

    evidence_ids: list[str] = Field(min_length=1)
    reason: str = Field(description="说明为什么需要重新启用这些 Evidence。")

    @field_validator("evidence_ids", mode="before")
    @classmethod
    def normalize_evidence_id_lists(cls, value: object) -> object:
        return normalize_evidence_ids(value)


class TaskComplete(BaseModel):
    """Researcher 提交任务结果并结束当前工具循环。"""

    reason: str = Field(description="为什么当前任务可以停止探索。")
    selected_evidence_ids: list[str] = Field(
        default_factory=list,
        max_length=EVIDENCE_REFERENCES_HARD_LIMIT,
        description="最终推荐给 Supervisor 的当前活跃 Evidence ID。",
    )

    @field_validator("selected_evidence_ids", mode="before")
    @classmethod
    def normalize_evidence_id_lists(cls, value: object) -> object:
        return normalize_evidence_ids(value)

    conclusion: str = Field(
        default="",
        description=(
            "任务级自然语言汇报，用连续文字一次说清当前状态：基于最终选中 Evidence "
            "综合出的事实、哪些局部问题已经查清、哪些仍未解决及原因、为什么在此停止。"
            "缺口写进这段叙述本身，不另开条目；不得引入未取证事实。"
        ),
    )
    finding: str = Field(
        default="",
        description=(
            "本方向查清了什么，供任务卡展示，两句话以内、不超过 120 字。"
            "只写结论性事实，不写检索与读取过程，不提未解决的缺口，不给后续建议，"
            "也不复述 conclusion 的首句。"
        ),
    )


class ResearchDelegate(BaseModel):
    """派发单个研究任务的工具调用 Schema；task_id 由本地程序分配，不信任模型。"""

    display_title: str = Field(
        min_length=2,
        max_length=TASK_TITLE_MAX_CHARS,
        description=(
            "任务卡上给人看的短标题：一个名词短语点明本任务要回答什么，"
            "不超过约 30 字；research_topic 的截断或复述不算标题。"
        ),
    )
    research_topic: str = Field(
        description=(
            "要研究的具体任务。必须包含研究对象、范围、待回答的局部问题、"
            "与已有任务的区别和完成标准；补缺时必须缩小到明确缺口，不能重述原问题。"
        )
    )


class WaitTask(BaseModel):
    """等待在途任务完成并收取任务报告；无在途时返回提示，不会卡死。"""

    task_ids: list[str] = Field(
        default_factory=list,
        description="要等待的任务 task_id 列表；留空表示等待任意一个在途任务完成。",
    )
    timeout: float | None = Field(
        default=None,
        ge=0.5,
        le=600,
        description="最长等待秒数；到期仍无任务完成返回 timed_out 回执。留空等到至少一个完成为止。",
    )

    @field_validator("task_ids", mode="before")
    @classmethod
    def strip_injected_whitespace_and_control_chars(cls, value: object) -> object:
        """剔除模型在长 id 流中夹入的空白与控制字符，避免 id 匹配不上在途。"""
        if not isinstance(value, list):
            return value
        cleaned = [re.sub(r"[\s\x00-\x1f\x7f]", "", str(item)) for item in value]
        return [item for item in cleaned if item]


class ResearchComplete(BaseModel):
    """冻结最新研究综合稿并终止研究阶段；版本对账由程序完成，不需要报版本号。"""

    reason: str = Field(description="为什么当前综合稿已经足以形成完整报告。")


class SynthesisFraming(BaseModel):
    """framing 操作的整值：只带要修改的字段，省略即不动；未知字段名直接拒绝。"""

    model_config = ConfigDict(extra="forbid")

    answer_goal: str | None = Field(default=None, max_length=STRUCTURED_TEXT_HARD_LIMIT_CHARS)
    thesis: str | None = Field(
        default=None,
        min_length=1,
        max_length=STRUCTURED_TEXT_HARD_LIMIT_CHARS,
        description=(
            "全文核心主张：一句话到一小段、可被 active Evidence 辩护的判断，"
            "各 aspect 为它服务或限定它。研究现状不需要在这里复述——工作台视图自带。"
        ),
    )
    decision_rationale: str | None = Field(
        default=None,
        max_length=STRUCTURED_TEXT_HARD_LIMIT_CHARS,
        description=(
            "本次变更各判断的根据：aspect 状态为何如此、证据绑定为何取舍、缺口为何入列。"
            "可分批补写，冻结验收时非空。"
        ),
    )

    @field_validator("answer_goal", "thesis", "decision_rationale", mode="before")
    @classmethod
    def empty_string_means_untouched(cls, value: object) -> object:
        """显式空串按"本字段不改"处理：空值不是内容，拒它只是白烧回合(实场 run-3234 T01 实证)。"""
        if isinstance(value, str) and not value.strip():
            return None
        return value


class WorkbenchUpsertAspect(BaseModel):
    """整值替换一个 aspect；身份只取 aspect.aspect_id。"""

    op: Literal["upsert_aspect"]
    aspect: ResearchAspect


class WorkbenchRemoveAspect(BaseModel):
    """删除一个 aspect。"""

    op: Literal["remove_aspect"]
    aspect_id: str = Field(min_length=1)


class WorkbenchUpdateFraming(BaseModel):
    """只改 framing 中给出的字段，省略即不动。"""

    op: Literal["framing"]
    framing: SynthesisFraming


class WorkbenchSetPlan(BaseModel):
    """plan 整列表替换。"""

    op: Literal["plan"]
    items: list[PlanItem] = Field(max_length=STRUCTURED_COLLECTION_HARD_LIMIT)


class WorkbenchMoveEvidence(BaseModel):
    """活跃证据槽位增量。"""

    op: Literal["evidence"]
    add: list[str] = Field(default_factory=list, max_length=STRUCTURED_COLLECTION_HARD_LIMIT)
    remove: list[str] = Field(default_factory=list, max_length=STRUCTURED_COLLECTION_HARD_LIMIT)


class WorkbenchAcknowledge(BaseModel):
    """工作台变动后确认现稿仍成立：只重盖 based_on 时间戳，不改稿。"""

    op: Literal["acknowledge"]


WorkbenchIntent = Annotated[
    Union[
        WorkbenchUpsertAspect,
        WorkbenchRemoveAspect,
        WorkbenchUpdateFraming,
        WorkbenchSetPlan,
        WorkbenchMoveEvidence,
        WorkbenchAcknowledge,
    ],
    Field(discriminator="op"),
]

WORKBENCH_INTENT_ADAPTER: TypeAdapter[WorkbenchIntent] = TypeAdapter(WorkbenchIntent)


def _normalize_changes(value: object) -> object:
    """修复模型把 changes 数组或其中单个操作再次编码成 JSON 字符串的常见偏差。

    嵌套越深模型越倾向把整个 changes 序列化成字符串；
    列表元素类型收紧后这种偏差会在工具调用框架层报错，到不了拒绝回执，
    所以在此处先行还原，让 services 的判别并集继续给出具名反馈。
    """
    if isinstance(value, str):
        raw = value.strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return [raw]
        return parsed if isinstance(parsed, list) else [parsed]
    if isinstance(value, list):
        normalized: list[object] = []
        for item in value:
            if isinstance(item, str):
                try:
                    normalized.append(json.loads(item))
                except json.JSONDecodeError:
                    normalized.append(item)
            else:
                normalized.append(item)
        return normalized
    return value


class UpdateWorkbench(BaseModel):
    """工作台唯一写入口：提交操作列表，每个操作整值替换其目标段。

    结构校验在此层保持宽松（changes 收原始对象），由 services 用判别并集解析后
    统一转成拒绝回执——工具调用层的校验错误到不了 rejections 的 did_you_mean 通道。
    """

    changes: list[Any] = Field(
        min_length=1,
        max_length=32,
        description=(
            '操作对象列表，每项带 op 判别字段：{"op":"upsert_aspect","aspect":{…完整 aspect}} | '
            '{"op":"remove_aspect","aspect_id":"…"} | {"op":"framing","framing":{…只带来改的字段}} | '
            '{"op":"plan","items":[…]} | {"op":"evidence","add":[…],"remove":[…]} | '
            '{"op":"acknowledge"}'
        ),
    )

    @field_validator("changes", mode="before")
    @classmethod
    def normalize_changes(cls, value: object) -> object:
        return _normalize_changes(value)


# 工具回执前缀：模型靠它区分工具结果与用户补充；observability 移除同一前缀
# 提取计数指标。三处共用，单一来源在此。
TOOL_RECEIPT_PREFIX = "【系统工具执行结果"


def format_tool_receipt(payload: object) -> str:
    """统一工具回执格式：前缀行 + JSON 串(或原样字符串)。"""
    body = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return f"{TOOL_RECEIPT_PREFIX}；非用户补充】\n{body}"


def _normalize_string_list(value: object) -> object:
    """修复模型把 JSON 字符串数组再次编码成字符串的常见偏差。"""
    if not isinstance(value, str):
        return value
    raw = value.strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        return parsed
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1].strip()
    # 兼容内部引号未转义、但数组元素分隔符仍明确的模型输出。
    parts = re.split(r'["”]\s*,\s*["“]', raw)
    return [item.strip().strip('"“”') for item in parts if item.strip().strip('"“”')]


class ReadEvidence(BaseModel):
    """读取目录中 Evidence 的原文与紧凑来源信息，以便引用可验证细节。"""

    evidence_ids: list[str] = Field(
        min_length=1,
        max_length=READ_EVIDENCE_REQUEST_WINDOW_IDS,
        description="要读取的 evidence_id 列表；引用前必须先读取。",
    )
    reason: str = Field(min_length=1)

    @field_validator("evidence_ids", mode="before")
    @classmethod
    def normalize_evidence_id_lists(cls, value: object) -> object:
        return normalize_evidence_ids(value)


class ReportComplete(BaseModel):
    """提交一篇带 Evidence 引用标记的 Markdown 草稿。"""

    title: str = Field(
        min_length=4,
        max_length=REPORT_TITLE_MAX_CHARS,
        description=(
            "为这篇回答拟的文章式标题：提纲挈领，可带文学性，但不夸张；"
            "纯文本，不含 # [ ] * 等 Markdown 标记与句读结尾。"
        ),
    )
    selected_evidence_ids: list[str] = Field(min_length=1)
    markdown: str = Field(min_length=1)

    @field_validator("selected_evidence_ids", mode="before")
    @classmethod
    def normalize_evidence_id_lists(cls, value: object) -> object:
        return normalize_evidence_ids(value)


class AskClarificationArgs(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=CLARIFICATION_QUESTION_MAX_CHARS,
        description=(
            "向用户提出的一个简短、具体问题，只解决一个会改变研究任务或研究边界的关键决策。"
            "不得询问能够通过检索自行回答的问题。"
        ),
    )
    options: list[str] = Field(
        min_length=CLARIFICATION_OPTION_COUNT,
        max_length=CLARIFICATION_OPTION_COUNT,
        description=(
            "恰好三个互斥、具体、可直接选择的默认答案；覆盖最可能的三种意图。"
            "不要加入‘其他/Other’，界面会自动提供自由输入。"
        ),
    )

    @field_validator("options", mode="before")
    @classmethod
    def normalize_options(cls, value: object) -> object:
        return _normalize_string_list(value)


class ClarificationCompleteArgs(BaseModel):
    headline: str = Field(
        min_length=2,
        max_length=RUN_HEADLINE_MAX_CHARS,
        description=(
            "把确认后的研究问题浓缩为一行短标题：名词性短语，不带句读结尾与 Markdown 标记；"
            "它是历史列表里给人看的题目，不是结论。"
        ),
    )
    intent_summary: str = Field(
        min_length=1,
        max_length=CLARIFICATION_INTENT_MAX_CHARS,
        description=(
            "对已确认用户意图和研究边界的准确摘要；不得改变或缩窄用户已经明确的要求。"
            "只写用户要什么、边界在哪，不规划研究重点——任务归 Supervisor。"
        ),
    )
    assumptions: list[str] = Field(
        default_factory=list,
        max_length=CLARIFICATION_ASSUMPTION_LIMIT,
        description="仍需由系统采用的显式假设，最多三项；没有假设时传空列表。",
    )

    @field_validator("assumptions", mode="before")
    @classmethod
    def normalize_string_lists(cls, value: object) -> object:
        return _normalize_string_list(value)
