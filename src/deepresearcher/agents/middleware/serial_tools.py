"""共享运行状态工具的串行执行控制。"""

from typing import Any, cast

from langchain.agents.middleware import AgentMiddleware

from deepresearcher.agents.middleware.concurrency import ToolExecutionGate


class SerialToolMiddleware(AgentMiddleware):
    """将状态边界工具作为独占屏障，其余工具允许共享并发。

    工具调度只经过读写栅栏 ToolExecutionGate:serial 工具取 exclusive,与所有
    取锁方互斥;其余取 shared,彼此可并发,但会被等待中的 exclusive 挡住,保持
    写优先。注册本中间件的 agent 其 LoopContext 必须携带 tool_gate,缺席当场抛
    AttributeError;Writer 没有 serial 工具,不注册本中间件。

    工具进不进 serial 名单依据行为而非名字:
    - exclusive:跨 await 的 check-then-act,如 Complete 读 revision 再冻结;
      或需要快照一致性的多结构读,如 ReadWorkingSet。
    - shared:并行 IO 读加 await 之后的纯追加写,如 ReadSources 写 documents
      和 source_refs;追加没有跨结构一致性问题,串行化只会损失吞吐。

    边界语义:Complete 等待的是同批中已经起跑的 delegate 与 shared 调用。同批
    tool_calls 按模型给出的列表序启动,Complete 排在前时等不到尚未启动的
    Delegate;[Complete, Delegate] 的顺序竞态是已知容忍项,出口有
    review→supervisor 的 freshness 再入环纠正,调度层不为此兜底;墙钟时间没有
    损失,同批本就由 ToolNode 汇合。

    删改 LoopContext 字段前先 grep 本文件。业务临界区另有按用途命名的业务锁,
    如 supervisor.bookkeeping_lock 与 researcher.commit_lock,与本调度无关。
    """

    def __init__(self, serial_tools: set[str]):
        super().__init__()
        self.serial_tools = serial_tools

    async def awrap_tool_call(self, request, handler):
        context = cast(Any, request.runtime.context)
        gate: ToolExecutionGate = context.tool_gate
        guard = (
            gate.exclusive() if request.tool_call["name"] in self.serial_tools else gate.shared()
        )
        async with guard:
            return await handler(request)
