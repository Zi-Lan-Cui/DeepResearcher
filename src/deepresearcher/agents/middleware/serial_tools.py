"""共享运行状态工具的串行执行控制。"""

from typing import Any, cast

from langchain.agents.middleware import AgentMiddleware

from deepresearcher.agents.middleware.concurrency import ToolExecutionGate


class SerialToolMiddleware(AgentMiddleware):
    """将状态边界工具作为独占屏障，其余工具允许共享并发。"""

    def __init__(self, serial_tools: set[str]):
        super().__init__()
        self.serial_tools = serial_tools

    async def awrap_tool_call(self, request, handler):
        # 唯一的调度机制:读写栅栏。context 契约(删改 LoopContext 字段前先 grep 本文件):
        #   入栅栏名单的判据(按此决定新工具进不进 serial,别看名字看行为):
        #     exclusive = 跨 await 的 check-then-act(如 Complete 读 revision 再冻结),
        #                 或需要快照一致的多结构读(如 ReadWorkingSet);
        #     shared    = 并行 IO 读+await 后的纯追加写(如 ReadSources 落 documents/
        #                 source_refs——追加操作无跨结构一致性问题,串行化只损吞吐);
        #     池级原子校验另走业务锁(commit_lock),那是"锁的名字说它护什么"的另一轨。
        #   tool_gate: ToolExecutionGate —— serial 工具取 exclusive(与一切互斥),
        #   其余取 shared(彼此可并发,但被 pending exclusive 挡住,写优先)。
        #   注册本中间件的 agent 其 LoopContext 必须携带 gate(下面直接属性读,
        #   缺席即 AttributeError);Writer 无 serial 工具、不注册本件,故不携带。
        #   业务临界区另有按用途命名的业务锁(supervisor.bookkeeping_lock /
        #   researcher.commit_lock),与调度无关。
        #   收尾语义(边界要说准):Complete(exclusive)等在飞的是**已起跑**的
        #   delegate/shared;同批 tool_calls 按模型写的列表序启动,Complete 排在前
        #   时栅栏无从"等未来"——[Complete, Delegate] 的顺序竞态属已知容忍
        #   (出口有 review→supervisor 的 freshness 再入环自纠),勿再往调度层
        #   上叠"包治"预期。墙钟不损失:同批本就被 ToolNode 汇合。
        context = cast(Any, request.runtime.context)
        gate: ToolExecutionGate = context.tool_gate
        guard = (
            gate.exclusive() if request.tool_call["name"] in self.serial_tools else gate.shared()
        )
        async with guard:
            return await handler(request)
