"""提示词外置层的守卫：每个运行时 prompt 都能加载且保留关键结构。"""

from deepresearcher.prompts import language_directive, load_prompt, render_data_section

ALL_PROMPTS = (
    "researcher",
    "supervisor",
    "clarifier",
    "writer",
    "router",
    "quick_answer",
    "reviewer",
    "language",
)


def test_every_prompt_loads_nonempty():
    for name in ALL_PROMPTS:
        assert load_prompt(name).strip(), f"{name}.md 为空/缺失"


def test_language_directive_matches_template_byte_for_byte():
    # 外置前后逐字一致：语言纪律模板必须原样产出。
    out = language_directive("中文")
    assert out.startswith("---\n\n## Output language") and out.endswith("leave code as is.\n")
    assert "{language}" not in out  # 占位符已被替换


def test_role_prompts_keep_identity_markers():
    assert "# Role" in load_prompt("researcher")
    assert "Supervisor" in load_prompt("supervisor")
    assert "overall reviewer" in load_prompt("reviewer")
    # writer 保留语言占位符（由调用点 .replace 填充）
    assert "__LANG__" in load_prompt("writer")


def test_researcher_prompt_teaches_early_evidence_submission_with_examples():
    prompt = load_prompt("researcher")

    assert "before the next search, page, or widening" in prompt
    assert "Do not call `AddEvidence` per sentence or per window" in prompt
    assert "never guesses and never trims characters on your behalf" in prompt
    assert prompt.count("### Example") == 3
    assert '"quote":"The system remained stable for 50 hours."' in prompt
    assert '"quote":"L18:' not in prompt


def test_supervisor_prompt_carries_dispatch_fewshots():
    prompt = load_prompt("supervisor")
    assert "## Examples" in prompt
    # 派发示例两字段各就各位，且反面说明禁止截断复述。
    assert '"display_title":' in prompt and '"research_topic":' in prompt
    assert "merely truncates" in prompt
    # 预算耗尽示例锁住 blocked 分支的正确动作。
    assert "round_budget_exhausted" in prompt


def test_prompts_keep_markdown_section_snapshot():
    expected_sections = {
        "clarifier": ("# Role", "## Decision tools", "## When to ask", "## Round budget"),
        "researcher": ("# Role", "## Scope", "## Searching and reading", "## Completion contract"),
        "supervisor": (
            "# Role",
            "## Scope",
            "## Tools",
            "## Budget and completion contract",
            "## Examples",
        ),
        "writer": ("# Role and boundary", "## Delivery principles", "## Writing quality"),
        "reviewer": ("# Role", "## Review target", "## Severity", "## Output boundary"),
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
