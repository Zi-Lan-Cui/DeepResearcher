"""串行工具的批次守卫:模型输出后校验工具批次,违规整批驳回、打回模型重新决策。

串行工具做跨结构一致读(如 ReadWorkingSet)或整字段快照写(如 Complete、Revise),
与同批任何其他调用并发都会让它读到不一致的现场。约束在模型输出之后、工具执行之前
(after_model)一次判定:发起消息带多个 tool_calls 且其中含串行工具时,整批驳回——
删除该 AIMessage、注入一条驳回 HumanMessage,jump 回模型重新决策。

整批拒绝保持回合原子性:不出现"一半工具跑了、另一半被拒"的半途现场,模型下一回合
看到的批次里没有任何工具执行过,按驳回说明完整重规划。单工具批与纯并行批不受影响。

判定只读 state 里已提交的发起消息,与工具执行和完成顺序无关,崩溃重放结论一致。

工具进不进串行名单依据行为:跨 await 的一致读或 replace 快照写才串行;纯 IO 读加追加
合并写(如 ReadSources、AddEvidence、SearchSources)并发安全,不进名单。
"""

from typing import Any

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage


class SerialToolMiddleware(AgentMiddleware):
    """串行工具必须独占一批:违规批次在模型输出后整批驳回,不执行任何工具。"""

    def __init__(self, serial_tools: set[str]):
        super().__init__()
        self.serial_tools = frozenset(serial_tools)

    @hook_config(can_jump_to=["model"])
    async def aafter_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        messages = state.get("messages", []) if isinstance(state, dict) else []
        last = messages[-1] if messages else None
        # 无 id 的发起消息无法定点撤销,放行比留下半撤销状态更安全(合成测试消息才会如此)。
        if not isinstance(last, AIMessage) or not last.tool_calls or last.id is None:
            return None
        calls = list(last.tool_calls)
        offenders = sorted({call["name"] for call in calls if call["name"] in self.serial_tools})
        if not offenders or len(calls) <= 1:
            return None
        content = (
            f"本次输出包含 {len(calls)} 个工具调用,其中 {', '.join(offenders)} 是串行工具,"
            "必须单独一轮调用;整批已驳回,没有任何工具执行。"
            "请重新规划:本轮只发出一个工具调用。"
        )
        return {
            "messages": [RemoveMessage(id=last.id), HumanMessage(content=content)],
            "jump_to": "model",
        }
