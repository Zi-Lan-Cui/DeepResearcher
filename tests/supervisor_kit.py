"""Supervisor 用例共用的装配件：模型替身、研究代理替身与轮次上下文构造函数。"""

import asyncio
import re

from langchain_core.messages import AIMessage, ToolMessage

from deepresearcher.agents.supervisor import Supervisor
from deepresearcher.agents.supervisor.state import (
    SupervisorContext,
    SupervisorRoundState,
)
from deepresearcher.config import AgentConfig
from deepresearcher.schemas import (
    PendingTask,
    ResearcherResult,
    TaskResult,
)
from fakes import evidence


class SupervisorLLM:
    """按消息历史无状态决策的模型替身，回合节奏与生产契约一致。"""

    def __init__(self, *, delegate_topics, complete_args=None, ready=False):
        self.delegate_topics = delegate_topics
        self.complete_args = complete_args
        self.ready = ready

    def bind_tools(self, _tools, tool_choice="any"):
        return self

    async def ainvoke(self, messages):
        history = "\n".join(str(message.content) for message in messages)
        revised = next(
            (
                message
                for message in reversed(messages)
                if isinstance(message, ToolMessage)
                and message.name == "UpdateWorkbench"
                and '"status": "applied"' in str(message.content)
                and '"synthesis_revision": 0' not in str(message.content)
            ),
            None,
        )
        if self.ready and revised is not None:
            return AIMessage(content="已建立可部分交付的综合稿，但尚未达到完整标准。")
        if self.complete_args is not None and revised is not None:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "ResearchComplete",
                        "args": {"reason": self.complete_args["reason"]},
                        "id": "call_complete",
                    }
                ],
            )
        has_wait_receipt = any(
            isinstance(message, ToolMessage) and message.name == "WaitTask" for message in messages
        )
        has_delegate_receipt = any(
            isinstance(message, ToolMessage) and message.name == "ResearchDelegate"
            for message in messages
        )
        if has_delegate_receipt and not has_wait_receipt:
            return AIMessage(
                content="",
                tool_calls=[{"name": "WaitTask", "args": {}, "id": "call_wait"}],
            )
        if (self.complete_args is not None or self.ready) and (
            has_wait_receipt or not self.delegate_topics
        ):
            evidence_ids = list(dict.fromkeys(re.findall(r'"evidence_id":\s*"([^"]+)"', history)))
            if evidence_ids:
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
                                            "thesis": "已有 Evidence 支撑核心结论。",
                                            "decision_rationale": (
                                                "已建立最小证据链。"
                                                if self.ready
                                                else self.complete_args["reason"]
                                            ),
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
                                            "conclusion": "直接回答问题",
                                            "evidence_ids": evidence_ids,
                                        },
                                    },
                                ]
                            },
                            "id": "call_revise",
                        }
                    ],
                )
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "ResearchComplete",
                        "args": {"reason": "fake 收尾理由。"},
                        "id": "call_complete",
                    },
                ],
            )
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "ResearchDelegate",
                    "args": {"display_title": "测试短题", "research_topic": topic},
                    "id": f"call_{index}",
                }
                for index, topic in enumerate(self.delegate_topics)
            ],
        )


class FixedResearcher:
    async def run(self, task):
        item = evidence(task["id"], task_id=task["id"])
        return ResearcherResult(
            evidences=[item],
            selected_evidence_ids=[item.evidence_id],
            source_refs=[item.source_url],
            task_result=TaskResult(
                task_id=task["id"],
                round=1,
                task_index=int(task.get("sequence", 1)),
                question=task["question"],
                research_direction=task["question"],
                execution_status="completed",
                coverage_status="sufficient",
                evidence_count=1,
                source_count=1,
                stop_reason="complete",
            ),
        )


COMPLETE_ARGS = {"reason": "两个方向均有直接来源。"}


class GatedResearcher(FixedResearcher):
    """完成时机由外部 Event 控制的研究代理替身，用于断言派发批不等待方向完成。"""

    def __init__(self) -> None:
        self.gate = asyncio.Event()
        self.entered: list[str] = []

    async def run(self, task):
        self.entered.append(task["id"])
        await self.gate.wait()
        return await super().run(task)


def delegate_context_for(agent, run_id: str = "run-g"):
    from deepresearcher.observability.scope import AgentScope

    config = AgentConfig(max_research_rounds=3, supervisor_max_concurrent_tasks=2)
    supervisor = Supervisor(
        SupervisorLLM(delegate_topics=["方向"], complete_args=None), config, research_agent=agent
    )
    round_state = SupervisorRoundState(
        {"query": "问题", "clarified_query": "问题", "evidences": []},
        current_round=1,
        active_evidence_limit=6,
    )
    context = SupervisorContext(
        scope=AgentScope(run_id=run_id, agent_name="Supervisor"),
        deps=supervisor._deps,
        round_state=round_state,
    )
    return supervisor, context


def pending_entry(task_id: str, question: str, title: str) -> dict:
    sequence = int(task_id.split("-")[1])
    task_record = {
        "id": task_id,
        "run_id": "run-g",
        "question": question,
        "display_title": title,
        "round": 1,
        "sequence": sequence,
        "worker_id": f"w-{task_id}",
        "worker_index": sequence,
        "parent_task_id": "",
        "operation_id": f"o-{task_id}",
    }
    return PendingTask(
        task_id=task_id, payload=task_record, display_title=title, round=1
    ).model_dump()
