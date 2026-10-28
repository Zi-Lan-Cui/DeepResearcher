"""工作台的运行时行为：批量变更、证据集合语义、草稿与视图。"""

import asyncio
import json
import re
from types import SimpleNamespace

from langchain_core.messages import AIMessage, ToolMessage

from deepresearcher.agents.supervisor import Supervisor, services
from deepresearcher.agents.supervisor.state import (
    SupervisorRoundState,
)
from deepresearcher.config import AgentConfig
from deepresearcher.schemas import (
    ResearchSynthesis,
    SynthesisFraming,
    TaskResult,
    TaskStopReason,
    UpdateWorkbench,
)
from fakes import make_evidence
from supervisor_kit import (
    FixedResearcher,
)


def _workbench_tool_messages(result: dict) -> list[ToolMessage]:
    return [
        message
        for message in result["supervisor_messages"]
        if isinstance(message, ToolMessage) and message.name == "UpdateWorkbench"
    ]


def test_workbench_rejects_evidence_outside_active_working_set():
    """Evidence 在档案但已释放时，引用它的批整单点名拒绝，跨键不变式在候选终态上判定。"""

    class ReleasedReferenceLLM:
        def bind_tools(self, _tools, tool_choice="any"):
            return self

        async def ainvoke(self, messages):
            history = "\n".join(str(message.content) for message in messages)
            evidence_ids = list(dict.fromkeys(re.findall(r'"evidence_id":\s*"([^"]+)"', history)))
            if "research_direction" not in history:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "ResearchDelegate",
                            "args": {
                                "display_title": "测试短题",
                                "research_topic": "证据将被释放的方向",
                            },
                            "id": "call_delegate",
                        }
                    ],
                )
            if not evidence_ids:
                return AIMessage(
                    content="",
                    tool_calls=[{"name": "WaitTask", "args": {}, "id": "call_wait"}],
                )
            if '"slot_removals"' not in history:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "UpdateWorkbench",
                            "args": {"changes": [{"op": "evidence", "remove": evidence_ids[:1]}]},
                            "id": "call_release",
                        }
                    ],
                )
            if '"invalid_evidence_ids"' not in history:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "UpdateWorkbench",
                            "args": {
                                "changes": [
                                    {
                                        "op": "framing",
                                        "framing": {
                                            "answer_goal": "回答测试问题",
                                            "thesis": "引用已释放 Evidence 的综合稿。",
                                            "decision_rationale": "故意引用非活跃 Evidence。",
                                        },
                                    },
                                    {
                                        "op": "upsert_aspect",
                                        "aspect": {
                                            "aspect_id": "core",
                                            "topic": "核心结论",
                                            "role": "主线",
                                            "required": True,
                                            "status": "covered",
                                            "conclusion": "仅有已释放的依据。",
                                            "evidence_ids": evidence_ids[:1],
                                        },
                                    },
                                ]
                            },
                            "id": "call_revise",
                        }
                    ],
                )
            return AIMessage(content="桌面已空且无法成稿,结束本轮。")

    supervisor = Supervisor(
        ReleasedReferenceLLM(),
        AgentConfig(max_research_rounds=1),
        research_agent=FixedResearcher(),
    )
    result = asyncio.run(
        supervisor.run(
            {
                "query": "研究问题",
                "clarified_query": "研究问题",
                "evidences": [],
            }
        )
    )

    batches = _workbench_tool_messages(result)
    assert len(batches) == 2  # 一单释放 applied、一单越权整单拒
    rejected = str(batches[1].content)
    assert '"status": "rejected"' in rejected
    assert '"invalid_evidence_ids"' in rejected
    assert "终态活跃证据" in rejected
    assert result["research_synthesis"] is None
    # 活跃集已空且没有落盘综合稿：不能进入写作。
    assert result["supervisor_next"] == "render_final_report"


def _apply_fixture() -> SupervisorRoundState:
    return SupervisorRoundState(
        {
            "query": "q",
            "evidences": [make_evidence("e1", "事实一"), make_evidence("e2", "事实二")],
            "active_evidence_ids": ["e1"],
        },
        current_round=1,
        active_evidence_limit=6,
    )


def test_workbench_repairs_double_encoded_changes_payload():
    """changes 被整体或单条二次编码成 JSON 字符串时，在 schema 层还原后照常走拒绝/施加管线。"""
    framing_op = {"op": "framing", "framing": {"thesis": "现状。"}}
    whole_array = UpdateWorkbench.model_validate(
        {"changes": json.dumps([framing_op, {"op": "acknowledge"}], ensure_ascii=False)}
    )
    assert all(isinstance(item, dict) for item in whole_array.changes)
    per_element = UpdateWorkbench.model_validate(
        {"changes": [json.dumps(framing_op, ensure_ascii=False), {"op": "acknowledge"}]}
    )
    assert isinstance(per_element.changes[0], dict)

    round_state = _apply_fixture()
    broken = services.apply_workbench_changes(round_state, ["not-json"])
    assert broken["status"] == "rejected"  # 无法还原的残串仍进拒绝回执，不触框架层
    assert "op" in str(broken["rejections"][0]["reason"])


def test_workbench_rejects_unknown_op_duplicates_and_malformed_aspect():
    round_state = _apply_fixture()

    unknown = services.apply_workbench_changes(
        round_state, [{"op": "reserve_evidence", "value": []}]
    )
    assert unknown["status"] == "rejected"
    assert "framing" in str(unknown["rejections"][0]["reason"])  # 可写清单随行告知

    framing_op = {
        "op": "framing",
        "framing": {
            "answer_goal": "目标",
            "thesis": "现状。",
            "decision_rationale": "理由。",
        },
    }
    duplicated = services.apply_workbench_changes(round_state, [framing_op, framing_op])
    assert duplicated["status"] == "rejected"
    assert duplicated["rejections"][0]["key"] == "framing"

    # aspect 身份只来自 value：键值双写不可表达，结构错误由 pydantic 逐字段点名。
    malformed = services.apply_workbench_changes(
        round_state, [{"op": "upsert_aspect", "aspect": {"aspect_id": "core"}}]
    )
    assert malformed["status"] == "rejected"
    assert all(
        issue["field"].startswith("upsert_aspect.aspect.")
        for issue in malformed["rejections"][0]["issues"]
    )
    assert round_state.research_synthesis is None  # 拒绝批不落盘


def test_workbench_plan_roundtrip_and_aspect_lifecycle():
    round_state = _apply_fixture()
    applied = services.apply_workbench_changes(
        round_state,
        [
            {
                "op": "plan",
                "items": [{"content": "核对来源", "status": "in_progress"}],
            },
            {
                "op": "framing",
                "framing": {
                    "answer_goal": "目标",
                    "thesis": "现状。",
                    "decision_rationale": "理由。",
                },
            },
            {
                "op": "upsert_aspect",
                "aspect": {
                    "aspect_id": "core",
                    "topic": "核心",
                    "role": "主线",
                    "status": "covered",
                    "conclusion": "s",
                    "evidence_ids": ["e1"],
                },
            },
        ],
    )
    assert applied["status"] == "applied"
    assert applied["synthesis_revision"] == 1
    assert round_state.plan == [{"content": "核对来源", "status": "in_progress"}]
    assert round_state.research_synthesis.selected_evidence_ids == ["e1"]

    removed = services.apply_workbench_changes(
        round_state, [{"op": "remove_aspect", "aspect_id": "core"}]
    )
    # 草稿期合法：aspect 删空、framing 仍在，稿子存在但过不了竣工验收。
    assert removed["status"] == "applied"
    assert round_state.research_synthesis is not None
    assert round_state.research_synthesis.aspects == []
    rejected_freeze = services.freeze_synthesis(round_state, "空块冻结")
    assert rejected_freeze["status"] == "rejected"
    assert "aspect" in rejected_freeze["rejection_cause"]

    ghost = services.apply_workbench_changes(
        round_state, [{"op": "remove_aspect", "aspect_id": "ghost"}]
    )
    assert ghost["status"] == "rejected"
    assert "known_aspect_ids" in str(ghost["rejections"][0])

    # plan 通道随 initial/from 往返。
    seeded = round_state.initial_channel_state()
    restored = SupervisorRoundState.from_channel_values(
        {"plan": seeded["plan"]},
        research_query="q",
        current_round=1,
        active_evidence_limit=6,
    )
    assert restored.plan == round_state.plan


def test_synthesis_draft_is_free_form_and_acceptance_deferred_to_freeze():
    """草稿期只要求单块 aspect；三件齐备在冻结时验收，缺谁点名谁；变更后全空清回未开稿。"""
    round_state = _apply_fixture()
    seeded = services.apply_workbench_changes(
        round_state,
        [
            {
                "op": "upsert_aspect",
                "aspect": {
                    "aspect_id": "core",
                    "topic": "核心",
                    "role": "主线",
                    "status": "covered",
                    "conclusion": "s",
                    "evidence_ids": ["e1"],
                },
            }
        ],
    )
    assert seeded["status"] == "applied"  # 无 framing 也能开稿
    assert round_state.research_synthesis.answer_goal == ""

    rejected = services.freeze_synthesis(round_state, "急着收尾")
    assert rejected["status"] == "rejected"
    cause = rejected["rejection_cause"]
    assert "answer_goal" in cause and "thesis" in cause and "decision_rationale" in cause

    completed = services.apply_workbench_changes(
        round_state,
        [
            {
                "op": "framing",
                "framing": {
                    "answer_goal": "目标",
                    "thesis": "这是全文要辩护的那一句。",
                    "decision_rationale": "依据。",
                },
            }
        ],
    )
    assert completed["status"] == "applied"
    assert services.freeze_synthesis(round_state, "验收通过")["status"] == "accepted"
    brief = services.report_brief_from_synthesis(round_state.completed_synthesis, max_caveats=6)
    assert brief.thesis == "这是全文要辩护的那一句。"

    cleared = services.apply_workbench_changes(
        round_state,
        [
            {"op": "remove_aspect", "aspect_id": "core"},
            {"op": "framing", "framing": {}},
        ],
    )
    assert cleared["status"] == "applied"  # 三件非空，稿子还在但再次过不了验收
    assert services.freeze_synthesis(round_state, "拆过头")["status"] == "rejected"

    legacy = ResearchSynthesis.model_validate(
        {
            "revision": 3,
            "based_on_working_set_revision": 2,
            "answer_goal": "旧稿目标",
            "overall_summary": "旧进度摘要被忽略",
            "thesis": "旧稿没有总纲,验收时补。",
            "aspects": [],
            "open_gaps": [],
            "conflicts": [],
            "decision_rationale": "旧依据。",
        }
    )
    assert legacy.thesis == "旧稿没有总纲,验收时补。"


def test_framing_explicit_empty_string_counts_as_untouched():
    """显式提交 thesis:"" 视作省略：稿子不动，版本不递增。"""
    framing = SynthesisFraming.model_validate({"thesis": "", "answer_goal": "   "})
    assert framing.thesis is None and framing.answer_goal is None

    round_state = _apply_fixture()
    services.apply_workbench_changes(
        round_state, [{"op": "framing", "framing": {"answer_goal": "目标"}}]
    )
    revision_before = round_state.research_synthesis.revision
    noop = services.apply_workbench_changes(
        round_state, [{"op": "framing", "framing": {"thesis": ""}}]
    )
    assert noop["status"] == "applied"
    assert round_state.research_synthesis.revision == revision_before
    assert round_state.research_synthesis.thesis == ""


def test_workbench_evidence_is_set_semantic_with_same_batch_slot_release():
    """add/remove 是集合语义：id 不得同时出现在两侧；同批腾出的槽位立即可用，回执与写库同源。"""
    round_state = SupervisorRoundState(
        {
            "query": "q",
            "evidences": [make_evidence("e1", "事实一"), make_evidence("e2", "事实二")],
            "active_evidence_ids": ["e1"],
        },
        current_round=1,
        active_evidence_limit=1,
    )

    conflict = services.apply_workbench_changes(
        round_state, [{"op": "evidence", "add": ["e2"], "remove": ["e2"]}]
    )
    assert conflict["status"] == "rejected"
    assert conflict["rejections"][0]["conflicting_evidence_ids"] == ["e2"]

    swap = services.apply_workbench_changes(
        round_state, [{"op": "evidence", "add": ["e2"], "remove": ["e1"]}]
    )
    assert swap["status"] == "applied"
    assert swap["slot_additions"] == ["e2"]
    assert swap["slot_removals"] == ["e1"]
    assert set(round_state.active_evidence_ids) == {"e2"}
    assert "skipped_restore_evidence_ids" not in swap


def test_workbench_batch_resolves_cross_op_dependencies_by_final_state():
    """同批操作间依赖按终态判定：remove 先于 upsert 折叠，结果与提交顺序无关。"""
    round_state = _apply_fixture()  # e1 活跃、e2 在档案
    built = services.apply_workbench_changes(
        round_state,
        [
            {
                "op": "framing",
                "framing": {
                    "answer_goal": "目标",
                    "thesis": "现状。",
                    "decision_rationale": "理由。",
                },
            },
            {
                "op": "upsert_aspect",
                "aspect": {
                    "aspect_id": "core",
                    "topic": "核心",
                    "role": "主线",
                    "status": "covered",
                    "conclusion": "s",
                    "evidence_ids": ["e1"],
                },
            },
        ],
    )
    assert built["status"] == "applied"

    restored_and_bound = services.apply_workbench_changes(
        round_state,
        [
            {"op": "evidence", "add": ["e2"]},
            {
                "op": "upsert_aspect",
                "aspect": {
                    "aspect_id": "core",
                    "topic": "核心",
                    "role": "主线",
                    "status": "covered",
                    "conclusion": "补回 e2 后同批绑定",
                    "evidence_ids": ["e1", "e2"],
                },
            },
        ],
    )
    assert restored_and_bound["status"] == "applied"
    assert "e2" in round_state.active_evidence_ids
    assert round_state.research_synthesis.selected_evidence_ids == ["e1", "e2"]

    churn = services.apply_workbench_changes(
        round_state,
        [
            {"op": "remove_aspect", "aspect_id": "core"},
            {
                "op": "upsert_aspect",
                "aspect": {
                    "aspect_id": "core",
                    "topic": "核心",
                    "role": "主线",
                    "status": "covered",
                    "conclusion": "删后重挂",
                    "evidence_ids": ["e1"],
                },
            },
        ],
    )
    assert churn["status"] == "applied"
    aspects = round_state.research_synthesis.aspects
    assert [aspect.aspect_id for aspect in aspects] == ["core"]
    assert aspects[0].conclusion == "删后重挂"


def test_workbench_acknowledge_restamps_without_editing_draft():
    round_state = _apply_fixture()
    built = services.apply_workbench_changes(
        round_state,
        [
            {
                "op": "framing",
                "framing": {
                    "answer_goal": "目标",
                    "thesis": "现状。",
                    "decision_rationale": "理由。",
                },
            },
            {
                "op": "upsert_aspect",
                "aspect": {
                    "aspect_id": "core",
                    "topic": "核心",
                    "role": "主线",
                    "status": "covered",
                    "conclusion": "s",
                    "evidence_ids": ["e1"],
                },
            },
        ],
    )
    assert built["status"] == "applied"
    frozen_before = round_state.research_synthesis
    assert frozen_before.based_on_working_set_revision == round_state.working_set_revision

    round_state._bump_working_set()  # 模拟：一波到达吸收了
    assert not round_state.synthesis_is_fresh(frozen_before)
    rejected_freeze = services.freeze_synthesis(round_state, "提前冻结")
    assert rejected_freeze["status"] == "rejected"

    ack = services.apply_workbench_changes(round_state, [{"op": "acknowledge"}])
    assert ack["status"] == "applied"
    assert round_state.research_synthesis.revision == 1  # 只重盖章，稿内容与版本不动
    accepted = services.freeze_synthesis(round_state, "表态后冻结")
    assert accepted["status"] == "accepted"

    eager = services.apply_workbench_changes(
        round_state,
        [
            {"op": "acknowledge"},
            {"op": "framing", "framing": {"thesis": "又改。"}},
        ],
    )
    assert eager["status"] == "rejected"  # 与实改同单时 acknowledge 冗余，拒绝引导单发


def test_workbench_view_shape_lock():
    round_state = _apply_fixture()
    from deepresearcher.agents.supervisor.state import workbench_view

    view = workbench_view(round_state)
    assert set(view) == {
        "workbench_revision",
        "plan",
        "inflight",
        "tasks",
        "active_evidence",
        "active_evidence_count",
        "active_evidence_limit",
        "reserve_evidence",
        "reserve_evidence_count",
        "flags",
        "synthesis_revision",
        "research_synthesis",
    }
    assert view["reserve_evidence"][0]["evidence_id"] == "e2"  # 未激活即 reserve 全卡
    assert [item["evidence_id"] for item in view["active_evidence"]] == ["e1"]
    assert view["flags"] == {"provider_exhausted": False, "rounds_left": 0}


def test_workbench_view_keeps_direction_conclusion_readable() -> None:
    """工作台是上下文压缩后的唯一档案：常态结论全文可见，极端超长才按句末收尾。"""
    from deepresearcher.agents.supervisor.state import workbench_view

    # 单句 28 字 × 120 ≈ 3,300 字，越过 2,000 上限才会触发截断。
    long_conclusion = "本方向由两条官方文档互证 Redis 的复制是异步的。" * 120
    state = SimpleNamespace(
        evidences=[],
        active_evidence_ids=set(),
        active_evidences=list,
        active_evidence_limit=6,
        working_set_revision=1,
        pending_tasks={},
        plan=[],
        provider_exhausted=False,
        research_synthesis=None,
        max_research_rounds=2,
        current_round=1,
        task_results=[
            TaskResult(
                task_id="task-0001",
                round=1,
                question="复制语义",
                research_direction="复制语义",
                display_title="复制语义",
                execution_status="completed",
                coverage_status="sufficient",
                evidence_count=0,
                source_count=0,
                conclusion=long_conclusion,
                stop_reason=TaskStopReason.COMPLETE,
            ),
            TaskResult(
                task_id="task-0002",
                round=1,
                question="持久化",
                research_direction="持久化",
                display_title="持久化",
                execution_status="completed",
                coverage_status="sufficient",
                evidence_count=0,
                source_count=0,
                conclusion="短结论。",
                stop_reason=TaskStopReason.COMPLETE,
            ),
        ],
    )
    tasks = workbench_view(state)["tasks"]
    clipped, intact = tasks[0]["conclusion"], tasks[1]["conclusion"]
    assert intact == "短结论。"
    assert clipped.endswith("…") and len(clipped) <= 2001
    assert clipped.removesuffix("…").endswith("。")  # 断在句末，不留在半句中间
    assert len(clipped) > 1900  # 上限 2000 字，实测最长 775 字的方向根本不触发
