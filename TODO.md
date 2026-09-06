# DeepSearch Agent TODO

> 当前阶段聚焦服务化交付质量、运行可靠性与降级策略，不累计已完成事项。
>
> 状态：`[x]` 已完成、`[~]` 部分完成、`[ ]` 待办。优先级：P0 必须先完成，P1 在主链稳定后完成，P2/P3 不阻塞交付。

## 当前判断

用户可见的产品闭环已经完成：提交研究、路由与澄清、等待用户回答、恢复执行、研究进度、报告交付、取消/重启恢复、历史列表与自适应分页均可用。当前不再有必须补齐的页面或主流程节点。

工程主线 [水平扩展 Worker 架构与迁移计划](docs/水平扩展Worker架构与迁移计划.md) M1–M8 已落地：API 与 Worker 可独立进程运行，多 Worker 共享 PostgreSQL 队列、lease、事件、usage 和缓存，LangGraph/Agent 核心逻辑未重写。

下一阶段只保留三个真正影响交付质量的方向：

1. **先收空转**：searcher 零良率熔断与候选池压力反馈；恢复期搜索/抓取/Evidence 缓存已完成。
2. **再做公网安全**：SSRF、接口限流、token 吊销和部署边界。
3. **最后做运维与体验增强**：集成 CI、数据保留、解释流、工具活动呈现和历史筛选。

## 当前基线（已完成）

- [x] 主流程：Router → Clarify → Supervisor → ResearchAgent → Writer（引用完整性校验）→ Reflection → Render。
- [x] Evidence 契约：claim/quote/来源/定位/确定性 quote 校验；`search_summary` 回退通道有 `support_ceiling=partial` 上限（摘要不升级为 direct）。
- [x] 领域模型与 State reducer、路由契约（NodeName/routing.py）、StopReason 唯一词汇、schemas 分域包化。
- [x] Agent 运行时中间件栈（profile/factory）：模型回合 + 工具 started/completed/failed + Agent 终止观测、提交守卫、模型/工具重试、ToolCallLimit、串行工具；writer 内联草稿救回。
- [x] Agent 运行身份与异步契约：四类 RuntimeContext 统一注入 `AgentExecutionScope`（run/agent/task/operation 归属）；所有 Agent `@tool` 统一使用原生 `async def`。
- [x] HTTP 韧性：`Retry-After` 感知退避 + 全抖动、provider 断路器、每供应商并发闸（`SEARCH_MAX_CONCURRENT_REQUESTS`）。
- [x] 服务化 P0：FastAPI + PostgreSQL（users/runs/run_events，SQLAlchemy async，NullPool/pragma 适配）+ 手写 SSE + 单文件前端；注册/登录（argon2 + HS256 JWT 12h）、每用户并发配额（429）、越权统一 404。
- [x] 事件链路：FanoutSink（seq 单点分配、pending backlog 补订阅、溢出丢旧+截断标记、ephemeral 旁路）+ CompositeSink + projector 默认拒绝白名单；断线/刷新从 RunEvent 回放，`run_done` 恒为收尾（孤儿 run 启动时补合成 done）。
- [x] 流式预览：官方 `astream(stream_mode=["values","messages"], subgraphs=True)` 由 `RunExecutor` 消费（嵌套 Pregel 的 token 经 ns 深度 1 归 supervisor 思考通道；ToolMessage 回执、tool_call 参数、非白名单通道全部隔离）；前端斜体预览 + 聚合帧"更长者胜"替换，永不出现打字完截短。
- [x] UI：阶段块（Router/Clarify/Supervisor/Writer/Reviewer/Render 事件驱动出现，绿呼吸点→灰/红收束）+ 方向卡（编号、滚动动作行、▸ 展开明细、状态配色）+ 全局时间线 + stats 指标条 + 论文式上标角标与来源面板互跳 + Clarifier 三选项/Other + 历史列表响应式宽度与 10/15/20 条自适应分页。
- [x] 语义修正：`completed ≠ 成功`（部分报告琥珀徽章）；reflection 对内容审查/坏 JSON 定向重试（`AGENT_REFLECTION_RETRY_ATTEMPTS`）；引用条数硬上限移除（曾制造两起 writer 死循环，聚焦交由提示词+审阅把关）。
- [x] writer 工具契约单一数字（读窗口 50 / 每轮交付 30 / 无引用上限），机制描述归工具、角色边界归系统提示词。
- [x] 输出语言运行时配置（`AGENT_OUTPUT_LANGUAGE`）注入全部六处生成用户可见文字的提示词；quote 保持原文例外。
- [x] 测试按引擎/服务/前端契约分主题组织；服务层测试跑文件 SQLite 还原 asyncpg 并发语义；测试进程显式使用与 Uvicorn 生产运行时一致的 uvloop，并有 LangChain model/tool wrapper 超时冒烟回归。
- [x] 真机端到端已验证：注册→提交→逐字流→报告→取消→kill -9 收敛→回放闭环→内容审查事故复盘。

## P0：当前必须收敛

### 当前版本封板

- [x] 本轮回归：Python 3.13 默认 asyncio loop 下 LangChain wrapper 静默挂起已用最小矩阵定位；测试对齐 Uvicorn 在 Linux 上的 uvloop 运行时，当前全量基线为 **280 passed / 3 infrastructure-gated skipped**。
- [x] 浏览器冒烟：10/15/20 条自适应分页、箭头/圆点换页、390px 窄屏无横向溢出、Clarifier 三选项 `等待回答 → 进行中`、服务重启后 `恢复续跑中` + 阶段回放 + `resuming` 事件，四条路径均已真机验证。
- [x] Worker 迁移 M1–M8：API 默认为纯控制面，`python -m deepsearch_agent.worker` 独立消费持久队列；多 Worker 原子 claim、lease 接管、恢复分诊选主、API 滚动重启回归均已落地。
- [x] M8 多进程故障/容量验收脚本：真实 Uvicorn + 2 Worker + PostgreSQL，自动覆盖 100 SSE、100 HTTP、1/2/4/8 并发、API 滚动替换、owner `SIGKILL` 后 `attempt=2` 接管与 `run_done` 唯一性；可控 Graph 零 LLM/provider 费用。

### G2：searcher 零良率空转熔断

- [ ] 方向内连续 N 次读取零新增证据 → 注入收敛指令（提示词要求基于现有证据交卷或如实上报不足），N 与文案待定（样本已足：三局 0 证据方向 4/4/1 个，事件文件可标定阈值）。
- [ ] 同方向重复检索的候选池压力反馈：SearchSources 结果文案随新候选率衰减。
- [ ] 内容审查拦截率的跨局统计（`reflection_retry` WARNING + `node_failed` 计数），决定是否补"审阅不可用"降级（approved-with-warning + 交付标注）。

验收：历史列表中 failed 且 0 证据的 run 显著减少；searcher 不再烧满 10 轮颗粒无收。

### 降级与恢复（四步走，进度 3/4）

- [x] ① checkpointer 基建（`e83dc4c`）：`build_graph(checkpointer=…)` + `thread_id=run_id`；服务 lifespan 挂 AsyncPostgresSaver（DSN 由业务 URL 派生，SQLite 自动跳过）；真机验证 quick run 落 9 行 checkpoint。
- [x] ② resume 驱动（`9839258`）：分诊式 reconcile + `resume_runs` 续跑 + PostgreSQL `Run.event_seq` 原子续号 + `resuming` 播报。**真机验收**：提交深度研究攒 3 断点后 `kill -9`，重启日志 `resuming_orphan_runs count=1`，续跑至自然完成（status=completed / report_rendered，run_done 恰好 1 帧，seq 5→919 无洞无撞）。
- [x] ③ 恢复期重复调用治理：PostgreSQL `ToolCache` 已覆盖 L1 规范化 query TTL、L2 canonical URL + parser/fetch version、L3 `content_hash + research_direction + extractor/model/schema/chunking version`。命中后仍产生当前 run/task 事件与 Evidence ID；失败、取消和部分 chunk 失败不写正向缓存。详见 [恢复期工具缓存方案](docs/恢复期工具缓存方案.md)。
- [x] ④ Clarifier 子图 + `interrupt()` + resume API（`awaiting_input` 状态、配额不占、三选项 + Other）——与②共用全部基建。
- [x] 产品入口收口为 Web：删除绕过持久 Run、用量、恢复和安全边界的 `main.py` CLI；人工 eval 与未来评测器统一通过 HTTP API 执行。

## P1：服务深化

### Service 职责与包结构收敛

> 先以当前调用链为基线补齐职责边界，再移动文件；不为了“看起来分层”而只做路径重命名。

- [x] 拆分 `RunManager` 的控制面与执行面职责：API 进程只保留 Run 创建/取消/恢复命令、持久事件发布和 checkpoint 可恢复性查询；Worker 进程持有 `RunExecutor`、LLM gates、HTTP client 与 Graph 组装。
- [x] 收口多 API 实例的准入竞态：`RunService.create()` 在 PostgreSQL 上用事务 advisory lock 串行化 count/insert，用户配额和全局 queued 上限对多 API 实例仍是原子的。
- [x] 将 `api.py` 拆为组装根、请求/响应 schema、鉴权依赖和按领域划分的 routes；`create_app()` 只负责 lifespan 与路由注册。
- [x] 按稳定职责将平铺的 `service/` 收纳为 `web/`、`runs/`、`execution/`、`events/`、`persistence/`；项目内导入已全部迁往唯一新路径，旧的纯 re-export 文件已删除。
- [x] 消除 PostgreSQL advisory lock 魔法数字：迁移、claim 容量、Worker 恢复和 API 准入 key 统一由 `service/coordination.py` 导出，并有稳定性回归。
- [x] 保持依赖方向 `web/control plane -> execution engine -> agents/tools`；纯 Usage ContextVar/预算异常已从 SQLAlchemy 服务存储中拆出，AST 架构测试阻止 Agent/LLM/Graph/Tool 反向导入 service。
- [x] 删除迁移期执行面代理：`RunManager` 不再持有 WorkerCoordinator、RunExecutor、RunWorker 或 tasks，embedded 模式由 lifespan 显式组装两个平面，仅注入 wake/cancel 窄回调。
- [x] 结构迁移批次均运行针对性回归，当前全量 280 passed / 3 infrastructure-gated skipped；M8 真实多进程验收通过 100 SSE、100 HTTP、1/2/4/8 并发、API 滚动替换及 Worker `SIGKILL` 后 attempt=2 接管。
- [ ] 定义有类型的 `ApiRuntime`，将分散的 `app.state.*` 收口为单一 lifespan 资源对象，减少 Web 层 `Any` 传播。
- [ ] 按责任拆分 `service/usage.py`：capacity/rate limiter 归 execution，UsageStore 归 persistence，LangChain callback 归 observability/integration。
- [~] 前端已从 1027 行单文件拆为语义 HTML、独立 CSS 与原生 ES module，静态资源有 HTTP 契约回归；后续随功能修改再按 api/state/sse/history/report 拆细模块，不为目录形式一次性重写稳定逻辑。

### 接口与数据

- [ ] detail API 返回渲染期有序引用（`ordered_citations` 含 `[来源N]` label），前端弃 markdown 反解析；报告头元信息（轮次/来源/Evidence）以字段下发。
- [ ] History 页补 stats 指标条（现仅详情页）与状态过滤（进行中/已完成/部分报告）；当前响应式列表和分页已完成。
- [ ] RunEvent 表保留/清理策略（随配额一起定 TTL）。
- [ ] writer 正文"打字"预览需改提交协议（正文走 content、cite 后校验）——评估收益后决定。

### 安全与运维（公网托管前逐项补齐）

- [x] **SSRF 守卫**：Researcher 的 fetch 请求仅允许 HTTP(S) 公网地址；拒绝凭据、私网、回环、链路本地与保留地址；逐跳校验重定向，并用 `CURLOPT_RESOLVE` 将连接固定到预检 DNS 结果以封闭重绑定窗口。
- [~] 请求限流：登录已使用 PostgreSQL 共享的账号/IP 双维度窗口与阻断期（HMAC 键、429 + `Retry-After`），多 API 实例不能换进程绕过；注册/创建 run 限流待补。token 吊销仍待 sessions 表或黑名单。
- [x] LLM/搜索/抓取用量与成本归集：`run_usage` 明细 + Run 聚合，actual/estimated 显式区分，详情 API 下发 token/费用/耗时/并发数据。
- [x] Alembic：`0001_initial`–`0004_run_usage`，应用启动自动 upgrade；旧库采纳与真实 PostgreSQL 迁移已验证。
- [ ] HTTPS/反代（Caddy 或 Nginx）与真实部署形态决策（BYO key 与否）。
- [ ] 健康检查：分离 `/health/live` 与 `/health/ready`；PostgreSQL/迁移影响 readiness，可丢失的 Redis 预览不作为业务就绪的硬条件。
- [ ] 制定 `run_events` / `run_usage` / tool cache / checkpoints / JSONL 的 TTL、分批清理、索引维护、备份与恢复演练策略。
- [ ] 生产密钥与连接边界：JWT secret 托管、PostgreSQL/Redis 私网或 TLS、按 API/Worker 总实例数核算连接池。
- [x] 进程边界机制化：同库可运行多 API/多 Worker，DB 分配 event seq，PostgreSQL NOTIFY 只做唤醒而非事实源。

### 引擎一致性

- [ ] LangGraph checkpoint serde 显式允许项目模型（至少 `RunLifecycle`、`NodeEvent`），并用 strict msgpack 回归旧断点恢复，避免未来版本默认阻断未注册类型。
- [ ] supervisor/researcher 提示词按 writer 同标准分层：工具机制下沉 description、系统提示词留角色/边界/输出标准。
- [ ] 并行派发观察：若模型持续每轮单派，评估程序层攒批并行（同 AIMessage 多 tool_calls 的能力已具备）。
- [ ] JSONL 事件轮转与大小上限；token/延迟计量入事件。

## P2：不阻塞交付

- [x] **Clarifier 恢复状态对齐**：用户提交回答后按 resume API 返回的真实 `queued` 显示“排队中”，仅在 Worker claim 后收到 `running` 事件时切换为“进行中”；系统重启恢复仍保留 `resuming`。
- [x] **跨进程 token 预览通道**：可选 Redis Pub/Sub 已接入 Worker 发布与 API SSE 合流；发送端和订阅端均使用有界丢旧队列。`text_delta` 无 seq、不落 RunEvent、不参与状态机，Redis 故障时自动退化为持久事件流。
- [ ] **Redis 预览连接复用**：当单 API 实例长期承载大量 SSE 连接时，将“每 SSE 一个 Pub/Sub 订阅”升级为进程内单读取器 + run_id 本地分发；在压测证明 Redis 连接数成为瓶颈时实施。
- [ ] **Router/Clarifier 解释流优化**：接入 `get_stream_writer()` + `stream_mode="custom"`；Router 在结构化校验完成后发送安全化 `reason`，Clarifier 开放正常文字预览，并在工具调用前解释判断依据。后端发送完整可信文本，逐字动画由前端完成，不用 `sleep()` 制造分片。
- [ ] **Clarifier 工具呈现优化**：`AskClarification` 的问题与三个选项保持原子渲染；`ClarificationComplete` 只提交最终结构化判断；增加“工具调用前说明理由”的守卫与空解释降级文案。
- [ ] **Supervisor 工具可视化**：为 `ResearchComplete`、`ResearchReady`、`ReadWorkingSet`、`ForgetEvidence` 补安全领域事件；区分永久阶段结论与低权重临时动作，不向前端暴露 Evidence ID、原始参数或异常详情。
- [ ] **排除方向智能渲染**：`delegate_completed(status=skipped|blocked)` 携带安全化方向摘要和标准原因（duplicate / budget / out_of_scope / covered），前端以灰色可折叠方向卡展示，不再统一压成“已跳过”。
- [ ] **统一工具活动协议**：评估 `tool_activity {stage, tool, phase, presentation}` 投影层，形成“模型解释 → 工具动作 → 权威阶段结论”的一致交互；custom 流只负责观感，持久化聚合事件负责回放与纠正。
- [ ] 跨 Run 语义检索与研究档案复用（先等短期上下文/checkpointer 落地）。
- [ ] 人工修订 Evidence/报告 + 版本审计；多项目/团队共享与权限隔离。
- [ ] eval/dataset.json 执行器与真实 LLM smoke 集；badcase 回归基线。
- [x] 基础 CI：GitHub Actions 已执行 locked sync、Ruff lint/format、Pyright、pytest 和 coverage。
- [ ] 集成 CI：增加真实 PostgreSQL/Redis service job 与 M8 多进程回归；HTML/PDF/DOCX 解析 fixture 继续扩充。
- [ ] 对象存储（PDF 导出、快照）与备份/隐私策略。

## 已知决策记录（避免反复）

- 引用条数不设硬上限（两起事故 vs 零保护价值）；聚焦度=提示词+审阅职责。
- 流式采用官方 `subgraphs=True`（曾自建 agent 调用点 relay，实测重复造轮后拆除，净删 121 行）。
- 预览通道 ephemeral：无 seq、不落库；聚合帧是唯一事实源，回放/重连以聚合帧收敛。
- 测试用文件 SQLite + NullPool 还原生产连接语义；aiosqlite 线程回调竞态以 filterwarnings 挂账（触发频率上升需重查）。
- `server.py` 与 `deepsearch_agent.worker` 都必须 `configure_logging()`；默认 API 不内嵌 Worker，本地单进程需显式打开兼容开关。
- `nodes/__init__` 的 reflection wrapper 遮蔽内核模块名：测试注入 `invoke_structured=` 须 import 内核函数。

## 文档维护规则

- [x] README 描述当前真实拓扑、配置与运行方式。
- [x] `docs/service-p0.md` 承载服务运行手册：启动/冒烟/状态语义/单进程纪律/明确不做清单与触发条件。
- [ ] 服务状态机、帧词表、降级策略变更时同步更新本文件与 runbook。
