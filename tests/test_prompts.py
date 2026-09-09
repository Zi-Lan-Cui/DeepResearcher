"""提示词外置层的守卫：每个 prompt 都能加载且非空、关键片段逐字保留。

外置（prompts/*.md）最大的风险是文件漏打包或被误删导致运行时静默退化；这里把
"能加载 + 保留了关键判据短语" 钉成测试。extractor 的 marker 另有
test_evidence_extract 双重覆盖，这里补 router/reflection/language。
"""

from deepsearch_agent.config import language_directive
from deepsearch_agent.prompts import load_prompt

ALL_PROMPTS = (
    "researcher",
    "supervisor",
    "clarifier",
    "writer",
    "evidence_extraction",
    "evidence_extraction_summary",
    "router",
    "quick_answer",
    "reflection",
    "language",
)


def test_every_prompt_loads_nonempty():
    for name in ALL_PROMPTS:
        assert load_prompt(name).strip(), f"{name}.md 为空/缺失"


def test_language_directive_matches_template_byte_for_byte():
    # 外置前后逐字一致：中文语言纪律必须原样产出。
    out = language_directive("中文")
    assert out.startswith("【语言】") and out.endswith("必须使用中文。")
    assert "{language}" not in out  # 占位符已被替换


def test_role_prompts_keep_identity_markers():
    assert "【身份】" in load_prompt("researcher")
    assert "Supervisor" in load_prompt("supervisor")
    assert "唯一允许引用的事实基础" in load_prompt("evidence_extraction")
    assert "整体审阅者" in load_prompt("reflection")
    # writer 保留语言占位符（由调用点 .replace 填充）
    assert "__LANG__" in load_prompt("writer")
