"""Supervisor 注入给模型的运行时指令文案。

与 ``prompts/supervisor.md`` 的区分：那份是常驻逐字加载的系统提示（网关前缀稳定），
这里是**按运行时条件才注入的短指令**——轮次预算耗尽、搜索提供方不可用、审阅回流。
把它们集中在此，控制流只负责装配结构化数据（status/reason/快照），措辞单独声明、
可 grep、可单测，改文案不必动编排逻辑。
"""

ROUND_BUDGET_EXHAUSTED_INSTRUCTION = (
    "研究轮次预算已耗尽；请修订最新研究综合稿。"
    "若达到完整标准则调用 ResearchComplete，否则直接结束，系统将按 partial 交付。"
)

PROVIDER_EXHAUSTED_INSTRUCTION = (
    "搜索服务账户级不可用（额度耗尽/密钥无效），系统性问题：再派新方向也会同样失败。"
    "停止派发 ResearchDelegate；把已有 Evidence 修订进综合稿，随后 "
    "ResearchComplete（足以成文）或 ResearchReady（保存部分报告）收尾。"
)

REVIEW_DECISION_RULES = {
    "rewrite": "Evidence 已覆盖核心问题，问题仅是措辞、范围、组织或已知材料利用不足；"
    "确认综合稿仍是最新版本后调用 ResearchComplete，进入改写。",
    "research": "核心结论缺少直接证据、来源矛盾，或必须补定义、比较对象或关键事实；"
    "调用 ResearchDelegate 补充方向。",
}
