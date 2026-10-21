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
    # 语言纪律模板必须原样产出，文案不漂移。
    out = language_directive("中文")
    assert out.startswith("---\n\n## Output language") and out.endswith("leave code as is.\n")
    assert "{language}" not in out


def test_role_prompts_keep_identity_markers():
    assert "# Role" in load_prompt("researcher")
    assert "Supervisor" in load_prompt("supervisor")
    assert "overall reviewer" in load_prompt("reviewer")
    # writer 保留语言占位符，由调用点 .replace 填充
    assert "__LANG__" in load_prompt("writer")


def test_researcher_prompt_teaches_early_evidence_submission_with_examples():
    prompt = load_prompt("researcher")

    assert "before the next search, page, or widening" in prompt
    assert "Do not call `AddEvidence` per sentence or per window" in prompt
    assert "never guesses and never strips markers on your behalf" in prompt
    assert "nearby_original_text" in prompt
    assert prompt.count("### Example") == 3
    assert '"quote_start":"The system remained","quote_end":"50 hours"' in prompt
    assert '"quote_start":"L18:' not in prompt
    assert "ambiguous_start" in prompt and "end_not_found" in prompt


def test_supervisor_prompt_carries_dispatch_fewshots():
    prompt = load_prompt("supervisor")
    assert "## Examples" in prompt
    # 派发示例两字段各就各位，反面说明禁止截断复述。
    assert '"display_title":' in prompt and '"research_topic":' in prompt
    assert "merely truncates" in prompt
    # 末轮示例校验结束动作：折叠、诚实 partial、完成或结束。
    assert "no further pass" in prompt and "honest partial status" in prompt
    # 不向模型描述程序侧不可达的防御分支。
    assert "status=blocked" not in prompt


def test_supervisor_prompt_documents_workbench_protocol():
    prompt = load_prompt("supervisor")
    assert "### `UpdateWorkbench`" in prompt
    assert '"op": "upsert_aspect"' in prompt
    assert '"op": "acknowledge"' not in prompt  # 操作表里 acknowledge 单列，示例不演示空转表态
    assert "acknowledge" in prompt
    assert "ResearchComplete` refuses" in prompt
    assert "ReviseResearchSynthesis" not in prompt
    assert "ReleaseEvidence" not in prompt


def test_prompts_keep_markdown_section_snapshot():
    expected_sections = {
        "clarifier": ("# Role", "## Decision tools", "## When to ask", "## Round budget"),
        "researcher": ("# Role", "## Scope", "## Searching and reading", "## Completion contract"),
        "supervisor": (
            "# Role",
            "## Scope",
            "## The workbench",
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
