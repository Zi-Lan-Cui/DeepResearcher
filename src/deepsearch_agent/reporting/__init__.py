"""报告文本管线：草稿引用协议的校验与最终报告的渲染。

Writer 侧只产出 evidence_id 键的草稿与绑定；编号（[[cite:id]] → [来源N]）
与参考来源表在审阅通过后的终检渲染层完成，且只发生一次。
"""

from deepsearch_agent.reporting.render import (
    no_evidence_blockers,
    render_error_report,
    render_final_report,
    render_incomplete_report,
)
from deepsearch_agent.reporting.validation import (
    extract_cite_ids,
    validate_and_bind,
)

__all__ = [
    "extract_cite_ids",
    "no_evidence_blockers",
    "render_error_report",
    "render_final_report",
    "render_incomplete_report",
    "validate_and_bind",
]
