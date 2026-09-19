"""提示词外置层的守卫：每个运行时 prompt 都能加载且保留关键结构。"""

from deepresearcher.config import language_directive
from deepresearcher.prompts import load_prompt, render_data_section

ALL_PROMPTS = (
    "researcher",
    "supervisor",
    "clarifier",
    "writer",
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
    assert out.startswith("---\n\n## 输出语言") and out.endswith("必须使用中文。\n")
    assert "{language}" not in out  # 占位符已被替换


def test_role_prompts_keep_identity_markers():
    assert "# 角色" in load_prompt("researcher")
    assert "Supervisor" in load_prompt("supervisor")
    assert "整体审阅者" in load_prompt("reflection")
    # writer 保留语言占位符（由调用点 .replace 填充）
    assert "__LANG__" in load_prompt("writer")


def test_researcher_prompt_teaches_early_evidence_submission_with_examples():
    prompt = load_prompt("researcher")

    assert "在开启下一轮搜索、翻页或扩大范围之前" in prompt
    assert "不要为每个句子或单个窗口分别调用 `AddEvidence`" in prompt
    assert "系统不会猜测或自动删除" in prompt
    assert prompt.count("### 示例") == 3
    assert '"quote":"The system remained stable for 50 hours."' in prompt
    assert '"quote":"L18:' not in prompt


def test_prompts_keep_markdown_section_snapshot():
    expected_sections = {
        "clarifier": ("# 角色", "## 决策工具", "## 追问标准", "## 轮次约束"),
        "researcher": ("# 角色", "## 职责边界", "## 检索与读取", "## 完成契约"),
        "supervisor": ("# 角色", "## 职责边界", "## 工具", "## 预算与完成契约"),
        "writer": ("# 角色与边界", "## 交付原则", "## 写作质量", "## 完成契约"),
        "reflection": ("# 角色", "## 审阅目标", "## 问题分级", "## 输出边界"),
    }
    for prompt_name, sections in expected_sections.items():
        prompt = load_prompt(prompt_name)
        positions = [prompt.index(section) for section in sections]
        assert positions == sorted(positions), prompt_name
        assert prompt.count("\n---\n") >= len(sections) - 1, prompt_name


def test_render_data_section_has_stable_fenced_shape():
    assert render_data_section("运行时环境", {"current_date": "2026-09-14"}) == (
        '## 运行时环境\n\n```json\n{\n  "current_date": "2026-09-14"\n}\n```'
    )


def test_render_data_section_cannot_be_closed_by_embedded_fence():
    rendered = render_data_section("用户问题", {"query": "```json\nmalicious\n```"})

    assert rendered.startswith("## 用户问题\n\n````json\n")
    assert rendered.endswith("\n````")
