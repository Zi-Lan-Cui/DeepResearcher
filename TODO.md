# DeepSearch Agent TODO

> 当前阶段聚焦服务化交付质量、运行可靠性与降级策略，不累计已完成事项。
>
> 状态：`[x]` 已完成、`[~]` 部分完成、`[ ]` 待办。优先级：P0 必须先完成，P1 在主链稳定后完成，P2/P3 不阻塞交付。

## 当前判断

用户可见的产品闭环已经完成：提交研究、路由与澄清、等待用户回答、恢复执行、研究进度、报告交付、取消/重启恢复、历史列表与自适应分页均可用。当前不再有必须补齐的页面或主流程节点。

下一阶段只保留三个真正影响交付质量的方向：

1. **先收运行成本与空转**：searcher 零良率熔断、搜索/抓取副作用幂等缓存。
2. **再做公网安全**：SSRF、接口限流、token 吊销和部署边界。
3. **最后做体验增强**：解释流、工具活动呈现、历史筛选等均为非阻塞优化。

## 当前基线（已完成）

- [x] 主流程：Router → Clarify → Supervisor → ResearchAgent → Writer（引用完整性校验）→ Reflection → Render。
- [x] Evidence 契约：claim/quote/来源/定位/确定性 quote 校验；`search_summary` 回退通道有 `support_ceiling=partial` 上限（摘要不升级为 direct）。
- [x] 领域模型与 State reducer、路由契约（NodeName/routing.py）、StopReason 唯一词汇、schemas 分域包化。
- [x] Agent 运行时中间件栈（profile/factory）：turn 日志、提交守卫、模型/工具重试、ToolCallLimit、串行工具；writer 内联草稿救回。
- [x] HTTP 韧性：`Retry-After` 感知退避 + 全抖动、provider 断路器、每供应商并发闸（`SEARCH_MAX_CONCURRENT_REQUESTS`）。
- [x] 服务化 P0：FastAPI + PostgreSQL（users/runs/run_events，SQLAlchemy async，NullPool/pragma 适配）+ 手写 SSE + 单文件前端；注册/登录（argon2 + HS256 JWT 12h）、每用户并发配额（429）、越权统一 404。
- [x] 事件链路：FanoutSink（seq 单点分配、pending backlog 补订阅、溢出丢旧+截断标记、ephemeral 旁路）+ CompositeSink + projector 默认拒绝白名单；断线/刷新从 RunEvent 回放，`run_done` 恒为收尾（孤儿 run 启动时补合成 done）。
- [x] 流式预览：官方 `astream(stream_mode=["values","messages"], subgraphs=True)` 在 RunManager 消费（嵌套 Pregel 的 token 经 ns 深度 1 归 supervisor 思考通道；ToolMessage 回执、tool_call 参数、非白名单通道全部隔离）；前端斜体预览 + 聚合帧"更长者胜"替换，永不出现打字完截短。
- [x] UI：阶段块（Router/Clarify/Supervisor/Writer/Reviewer/Render 事件驱动出现，绿呼吸点→灰/红收束）+ 方向卡（编号、滚动动作行、▸ 展开明细、状态配色）+ 全局时间线 + stats 指标条 + 论文式上标角标与来源面板互跳 + Clarifier 三选项/Other + 历史列表响应式宽度与 10/15/20 条自适应分页。
- [x] 语义修正：`completed ≠ 成功`（部分报告琥珀徽章）；reflection 对内容审查/坏 JSON 定向重试（`AGENT_REFLECTION_RETRY_ATTEMPTS`）；引用条数硬上限移除（曾制造两起 writer 死循环，聚焦交由提示词+审阅把关）。
- [x] writer 工具契约单一数字（读窗口 50 / 每轮交付 30 / 无引用上限），机制描述归工具、角色边界归系统提示词。
- [x] 输出语言运行时配置（`AGENT_OUTPUT_LANGUAGE`）注入全部六处生成用户可见文字的提示词；quote 保持原文例外。
- [x] 测试按引擎/服务/前端契约分主题组织；服务层测试跑文件 SQLite 还原 asyncpg 并发语义。
- [x] 真机端到端已验证：注册→提交→逐字流→报告→取消→kill -9 收敛→回放闭环→内容审查事故复盘。

## P0：当前必须收敛

### 当前版本封板

- [x] 本轮回归：`test_service_runs.py` 的“挂起”已定位为受限沙箱阻断 aiosqlite 工作线程回调，非业务死锁；改在非沙箱环境并禁用外部 pytest 插件自动加载后，全量测试 219 条通过。
- [ ] 浏览器冒烟：验证 10/15/20 条自适应分页、窄屏布局、Clarifier 提问→回答→继续、服务重启自动恢复四条路径。

### G2：searcher 零良率空转熔断

- [ ] 方向内连续 N 次读取零新增证据 → 注入收敛指令（提示词要求基于现有证据交卷或如实上报不足），N 与文案待定（样本已足：三局 0 证据方向 4/4/1 个，事件文件可标定阈值）。
- [ ] 同方向重复检索的候选池压力反馈：SearchSources 结果文案随新候选率衰减。
- [ ] 内容审查拦截率的跨局统计（`reflection_retry` WARNING + `node_failed` 计数），决定是否补"审阅不可用"降级（approved-with-warning + 交付标注）。

验收：历史列表中 failed 且 0 证据的 run 显著减少；searcher 不再烧满 10 轮颗粒无收。

### 降级与恢复（四步走，进度 3/4）

- [x] ① checkpointer 基建（`e83dc4c`）：`build_graph(checkpointer=…)` + `thread_id=run_id`；服务 lifespan 挂 AsyncPostgresSaver（DSN 由业务 URL 派生，SQLite 自动跳过）；真机验证 quick run 落 9 行 checkpoint。
- [x] ② resume 驱动（`9839258`）：分诊式 reconcile + `resume_runs` 续跑 + `seed_seq` 跨世续号 + `resuming` 播报。**真机验收**：提交深度研究攒 3 断点后 `kill -9`，重启日志 `resuming_orphan_runs count=1`，续跑至自然完成（status=completed / report_rendered，run_done 恰好 1 帧，seq 5→919 无洞无撞）。
- [ ] ③ 副作用幂等：搜索/抓取结果缓存（content hash、query 去重——即 P1-A 旧账），否则每次恢复重付断点节点的钱。
- [x] ④ Clarifier 子图 + `interrupt()` + resume API（`awaiting_input` 状态、配额不占、三选项 + Other）——与②共用全部基建。
- [ ] `build_graph()` 显式持有/关闭依赖的 CLI 侧对齐（服务侧 lifespan 已做；CLI 仍在 main.py 手工组装）。

## P1：服务深化

### 接口与数据

- [ ] detail API 返回渲染期有序引用（`ordered_citations` 含 `[来源N]` label），前端弃 markdown 反解析；报告头元信息（轮次/来源/Evidence）以字段下发。
- [ ] History 页补 stats 指标条（现仅详情页）与状态过滤（进行中/已完成/部分报告）；当前响应式列表和分页已完成。
- [ ] RunEvent 表保留/清理策略（随配额一起定 TTL）。
- [ ] writer 正文"打字"预览需改提交协议（正文走 content、cite 后校验）——评估收益后决定。

### 安全与运维（公网托管前逐项补齐）

- [ ] **SSRF 守卫**：fetcher 拦截私网/回环/云元数据地址 + DNS 重绑定防护——一票否决项。
- [ ] 请求限流（登录/注册/创建 run）；token 吊销（sessions 表或黑名单）。
- [ ] LLM/搜索调用的 token 用量与成本归集（事件已有阶段，缺 usage 字段）。
- [ ] Alembic（触发条件=第一次 schema ALTER，当前 create_all+追加列纪律）。
- [ ] HTTPS/反代（Caddy 或 Nginx）与真实部署形态决策（BYO key 与否）。
- [ ] 单进程纪律的机制化（当前仅文档约定：同库禁双 server，fanout/seq 为进程内存）。

### 引擎一致性

- [ ] supervisor/researcher 提示词按 writer 同标准分层：工具机制下沉 description、系统提示词留角色/边界/输出标准。
- [ ] 并行派发观察：若模型持续每轮单派，评估程序层攒批并行（同 AIMessage 多 tool_calls 的能力已具备）。
- [ ] JSONL 事件轮转与大小上限；token/延迟计量入事件。

## P2：不阻塞交付

- [ ] **Router/Clarifier 解释流优化**：接入 `get_stream_writer()` + `stream_mode="custom"`；Router 在结构化校验完成后发送安全化 `reason`，Clarifier 开放正常文字预览，并在工具调用前解释判断依据。后端发送完整可信文本，逐字动画由前端完成，不用 `sleep()` 制造分片。
- [ ] **Clarifier 工具呈现优化**：`AskClarification` 的问题与三个选项保持原子渲染；`ClarificationComplete` 只提交最终结构化判断；增加“工具调用前说明理由”的守卫与空解释降级文案。
- [ ] **Supervisor 工具可视化**：为 `ResearchComplete`、`ResearchReady`、`ReadWorkingSet`、`ForgetEvidence` 补安全领域事件；区分永久阶段结论与低权重临时动作，不向前端暴露 Evidence ID、原始参数或异常详情。
- [ ] **排除方向智能渲染**：`delegate_completed(status=skipped|blocked)` 携带安全化方向摘要和标准原因（duplicate / budget / out_of_scope / covered），前端以灰色可折叠方向卡展示，不再统一压成“已跳过”。
- [ ] **统一工具活动协议**：评估 `tool_activity {stage, tool, phase, presentation}` 投影层，形成“模型解释 → 工具动作 → 权威阶段结论”的一致交互；custom 流只负责观感，持久化聚合事件负责回放与纠正。
- [ ] 跨 Run 语义检索与研究档案复用（先等短期上下文/checkpointer 落地）。
- [ ] 人工修订 Evidence/报告 + 版本审计；多项目/团队共享与权限隔离。
- [ ] eval/dataset.json 执行器与真实 LLM smoke 集；badcase 回归基线。
- [ ] CI（lint→typecheck→unit→integration）；HTML/PDF/DOCX 解析 fixture 扩充。
- [ ] 对象存储（PDF 导出、快照）与备份/隐私策略。

## 已知决策记录（避免反复）

- 引用条数不设硬上限（两起事故 vs 零保护价值）；聚焦度=提示词+审阅职责。
- 流式采用官方 `subgraphs=True`（曾自建 agent 调用点 relay，实测重复造轮后拆除，净删 121 行）。
- 预览通道 ephemeral：无 seq、不落库；聚合帧是唯一事实源，回放/重连以聚合帧收敛。
- 测试用文件 SQLite + NullPool 还原生产连接语义；aiosqlite 线程回调竞态以 filterwarnings 挂账（触发频率上升需重查）。
- 同库单 server 进程；`server.py` 必须 `configure_logging()`（否则 lastResort 吞服务层 INFO）。
- `nodes/__init__` 的 reflection wrapper 遮蔽内核模块名：测试注入 `invoke_structured=` 须 import 内核函数。

## 文档维护规则

- [x] README 描述当前真实拓扑、配置与运行方式。
- [x] `docs/service-p0.md` 承载服务运行手册：启动/冒烟/状态语义/单进程纪律/明确不做清单与触发条件。
- [ ] 服务状态机、帧词表、降级策略变更时同步更新本文件与 runbook。
