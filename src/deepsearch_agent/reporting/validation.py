"""草稿引用协议校验：以 evidence_id 为键产出绑定与引用元数据，不渲染编号。"""

import re

from deepsearch_agent.errors import AgentError
from deepsearch_agent.evidence.models import Evidence
from deepsearch_agent.schemas import Citation, ParagraphBinding

_FENCED_CODE = re.compile(
    r"(^|\n)(?P<fence>`{3,}|~{3,})[^\n]*\n.*?\n(?P=fence)(?=\n|$)",
    re.MULTILINE | re.DOTALL,
)
_INLINE_CODE = re.compile(r"(?P<tick>`+).*?(?P=tick)", re.DOTALL)
_CITE_MARKER = re.compile(r"\[\[cite:(?P<sources>[^\]\r\n]+)\]\]", re.IGNORECASE)
_CITATION_MARKER = re.compile(r"\[来源(?P<index>\d+)\]")


class DraftProtocolError(AgentError, ValueError):
    """Writer 草稿的本地引用协议不成立，可要求 Writer 定向修复。"""

    code = "writer_protocol"


def extract_cite_ids(markdown: str) -> set[str]:
    """读取句末引用标记的 Evidence ID，忽略代码示例中的伪标记。"""
    without_fences = _FENCED_CODE.sub("", markdown)
    without_code = _INLINE_CODE.sub("", without_fences)
    ids: set[str] = set()
    for match in _CITE_MARKER.finditer(without_code):
        ids.update(
            item.strip().strip("[]")
            for item in re.split(r"[,，、]", match.group("sources"))
            if item.strip()
        )
    return ids


def validate_and_bind(
    markdown: str,
    evidence_by_id: dict[str, Evidence],
) -> tuple[str, list[ParagraphBinding], list[Citation]]:
    """校验草稿的引用协议，产出 evidence_id 键的正文、绑定与引用元数据。

    正文保留 [[cite:evidence_id]] 内部标记；编号渲染在审阅通过后的
    终检渲染层完成。任何协议违反都抛 DraftProtocolError，由 Writer
    在自身有限重试内修复。
    """
    if re.search(r"^#{1,6}\s*参考来源\b", markdown, re.MULTILINE):
        raise DraftProtocolError("正文不得自行生成‘参考来源’小节。")
    _reject_manual_markers(markdown)
    bindings = _extract_bindings(markdown)
    cited = list(dict.fromkeys(source_id for item in bindings for source_id in item.evidence_ids))
    if not cited:
        raise DraftProtocolError("正文没有有效 cite 标签。")
    unknown = [source_id for source_id in cited if source_id not in evidence_by_id]
    if unknown:
        raise DraftProtocolError(f"cite 使用了不存在的 Evidence：{', '.join(unknown)}。")
    citations = [_citation(evidence_by_id[source_id]) for source_id in cited]
    return markdown.strip(), bindings, citations


def _reject_manual_markers(markdown: str) -> None:
    """拒绝手写编号、错误 cite 格式与已废弃标签；不触碰代码块与行内代码。"""
    without_fences = _FENCED_CODE.sub("", markdown)
    without_code = _INLINE_CODE.sub("", without_fences)
    # 先移除全部合法 [[cite:...]] 标记，再检查剩余文本中的违规协议。
    text_outside_cites = _CITE_MARKER.sub("", without_code)
    if _CITATION_MARKER.search(text_outside_cites):
        raise DraftProtocolError(
            "不要手动写 [来源N]；请使用 [[cite:evidence_id]]，由本地程序统一编号。"
        )
    if "[[cite:" in text_outside_cites.lower():
        raise DraftProtocolError("cite 标记格式错误；必须写成完整的 [[cite:evidence_id]]。")
    if re.search(r"</?cite\b", text_outside_cites, re.IGNORECASE):
        raise DraftProtocolError("不再支持 <cite> 标签；请使用句末 [[cite:evidence_id]] 标记。")


def _extract_bindings(markdown: str) -> list[ParagraphBinding]:
    """把草稿拆成可审阅块：剥掉 [[cite:...]] 标记，保留整段论证文本。"""
    bindings: list[ParagraphBinding] = []

    def add_block(block: str) -> None:
        block = block.strip()
        if not block or re.fullmatch(r"#{1,6}\s+.*", block):
            return
        # 引用标记出现在代码中不属于正文论断；行内代码同样不能形成绑定。
        citation_view = _INLINE_CODE.sub("", block)
        source_ids: list[str] = []
        for match in _CITE_MARKER.finditer(citation_view):
            raw_ids = [
                source_id.strip().strip("[]")
                for source_id in re.split(r"[,，、]", match.group("sources"))
                if source_id.strip()
            ]
            if len(raw_ids) > 3:
                raise DraftProtocolError("单个 cite 标记最多绑定三个 Evidence。")
            source_ids.extend(raw_ids)
        source_ids = list(dict.fromkeys(source_ids))
        text = _CITE_MARKER.sub("", block)
        text = _INLINE_CODE.sub("", text).strip()
        if not text:
            return
        kind = (
            "transition" if not source_ids else ("synthesis" if len(source_ids) > 1 else "evidence")
        )
        bindings.append(ParagraphBinding(text=text, kind=kind, evidence_ids=source_ids))

    def add_prose(prose: str) -> None:
        for paragraph in re.split(r"\n\s*\n", prose):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            lines = paragraph.splitlines()
            # 表格以行作为最小可引用单元；标题分隔线没有论断。
            if len(lines) > 1 and all(line.lstrip().startswith("|") for line in lines):
                for line in lines:
                    if not re.fullmatch(r"\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*", line):
                        add_block(line)
                continue
            # 列表项通常承载独立事实或论点，也不能合并为一个引用范围。
            if all(re.match(r"\s*(?:[-*+] |\d+[.)] )", line) for line in lines):
                for line in lines:
                    add_block(line)
                continue
            add_block(paragraph)

    cursor = 0
    for fenced in _FENCED_CODE.finditer(markdown):
        add_prose(markdown[cursor : fenced.start()])
        cursor = fenced.end()
    add_prose(markdown[cursor:])
    return bindings


def _citation(evidence: Evidence) -> Citation:
    """以 evidence_id 为键构造可审计引用元数据；显示编号由渲染层分配。"""
    return Citation(
        id=evidence.evidence_id,
        url=evidence.source_url,
        title=evidence.source_title,
        quote=evidence.quote,
        claim=evidence.claim,
        source_profile=evidence.source_profile,
    )
