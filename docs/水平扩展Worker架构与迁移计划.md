# DeepSearch 水平扩展 Worker 架构与迁移计划

## 1. 目标与非目标

目标是将当前“FastAPI 进程受理 Run，同一进程内 `asyncio.create_task()` 执行 LangGraph”的形态，逐步演进为 API 与多个 Worker 可独立伸缩的形态。迁移期间必须保持：

- Router → Clarifier → Supervisor/Researcher → Writer/Reflection 的图逻辑不改。
- `thread_id=run_id` 与 PostgreSQL Checkpointer 的恢复语义不改。
- API、SSE、取消、澄清恢复和历史报告在每个迁移步骤都可用。
- 单进程部署始终是可用的默认形态，不为尚未需要的分布式运维付出成本。

非目标：本迁移不重写 Agent，不将 LangGraph State 放入 Redis，不立即引入 Celery/Kubernetes，不追求 exactly-once。底层队列和 checkpoint 均采用 at-least-once，通过租约、状态 CAS 和工具语义缓存实现可安全重放。

## 2. 为什么现状无法直接增加 Worker

当前 `RunManager` 同时是受理服务、调度器、执行器、事件排水器和停机管理器。其中以下数据只存在当前 Python 进程：

- `_tasks`：Run 到 `asyncio.Task` 的映射。
- `_persisted/_done_published/_shutdown_interrupts`：终态、done 和停机竞态防护。
- `FanoutSink._counters/_pending/_subs`：事件 seq、未落库事件和 SSE 订阅者。
- Agent/Search/Evidence 内的 `asyncio.Semaphore`：只限制单个 Graph/Run，不限制整个集群。
- 内存搜索缓存与供应商断路时间窗。

直接启动多个 Uvicorn Worker 会导致同一 Run 被重复恢复、seq 冲突、跨进程取消失效、SSE 收不到另一进程事件，以及供应商并发上限被 Worker 数成倍放大。

解决方式不是将这些数据搬到另一个“管理进程”的内存，而是将决定正确性的状态放入 PostgreSQL，将可丢失的低延迟通知通过可替换的 notifier 传播。

## 3. 目标架构与职责

```text
Browser
   │ HTTP / SSE
Load Balancer
   │
API instances
   ├─ RunService: create/cancel/resume/query/quota
   └─ EventStream: DB replay + notifier live tail
             │
             ├─ PostgreSQL
             │    ├─ runs + queue/lease
             │    ├─ run_events + usage
             │    └─ LangGraph checkpoints
             │
             └─ EventNotifier (local → PG NOTIFY/Redis)
                         │
Worker instances          │
   ├─ RunScheduler/RunQueue.claim()
   ├─ RunExecutor
   │    └─ build_graph().astream()
   └─ lease heartbeat / cancellation probe
```

### RunService（控制面）

负责用户意图和业务规则：创建 Run、每用户 queued/running 配额、取消意图、提交澄清答案、查询。它不构建 Graph，不保存 `asyncio.Task`，不选择具体 Worker。

### RunQueue / RunScheduler（调度面）

负责持久排队、全局执行槽、优先级、公平性、原子 claim、租约与过期回收。调度器可以崩溃，但任务不能丢，因此队列事实在 PostgreSQL，本地 `asyncio.Event` 只用于减少轮询延迟。

### RunExecutor（执行面）

只执行一个已领取 Run：装配 Graph、调用 `astream`、发布预览、处理 interrupt、保存终态/部分报告、排水事件。它不检查用户队列配额，不决定下一个 Run。

### EmbeddedWorker 与独立 Worker

迁移初期在 API lifespan 内启动 EmbeddedWorker，对外仍是单进程。当所有共享状态完成外移后，只新增 Worker 启动入口并关闭 API 内的 embedded 模式，Graph 代码不改。

## 4. Run 状态机与队列语义

```text
                 claim + lease
queued  ------------------------------> running
  │                                         │
  │ cancel                                  ├─> awaiting_input
  v                                         │        │ answer persisted
cancelled                                   │        v
                                            │      queued
                                            ├─> completed
                                            ├─> failed
                                            ├─> cancelled
                                            └─> interrupted/queued
                                                 lease expired
```

- `queued` 就是 Run 级等待态，不增加 `pending`，避免与 SubTask/Review 的 pending 混淆。
- `running` 必须同时有未过期 lease；运行行不是 Worker 所有权的充分证明。
- `awaiting_input` 不占执行槽。答案持久化后重新入队，Worker 以 `Command(resume=...)` 继续原 thread。
- `interrupted` 是可恢复事实；调度层将其转成可 claim 项，而不由所有 API 实例同时扫描后起任务。
- 终态只能通过带期望状态和 lease owner 的 CAS 更新写入，不再依赖进程内 `_persisted` 作为最终保障。

建议增加字段：`priority`、`available_at`、`worker_id`、`lease_expires_at`、`heartbeat_at`、`execution_attempt`、`cancellation_requested_at`、`resume_payload`、`queue_entered_at`、`queue_wait_ms`。这是第一次正式 ALTER，必须同时引入 Alembic，不再仅依赖 `create_all()`。

## 5. 原子领取、租约和故障恢复

PostgreSQL 实现使用短事务的 `SELECT ... FOR UPDATE SKIP LOCKED`，或等价的 CTE `UPDATE ... RETURNING`。claim 在同一事务内完成 `queued → running`、写入 owner/lease 和 `execution_attempt + 1`。

Worker 每隔固定时间续租，续租 SQL 必须带 `run_id + worker_id + status=running`。续租失败表示所有权丢失，Worker 必须取消本地执行，不能继续写终态。

回收器将租约过期且非终态的 Run 转为 `interrupted` 后重新入队。新 Worker 先确认 checkpoint 存在，再使用 `astream(None)`；无 checkpoint 才写 `failed/server_restart`。这保留现有 resume 引擎，只将“谁启动 resume”从 API 进程换成 Scheduler/Worker。

## 6. 取消与澄清恢复迁移

### 取消

API 原子写 `cancellation_requested_at`；queued/awaiting_input 可直接转 cancelled，running 则保留执行所有权，由 owner Worker 在取消 probe 观察到意图后 `task.cancel()` 并写终态。通知可先用 PostgreSQL 轮询/LISTEN NOTIFY，后续换 Redis Pub/Sub；即使通知丢失，数据库意图仍会被 probe 发现。

### 澄清

`interrupt()` 与 checkpoint 不变。Worker 写 awaiting_input 并释放 lease。API 接受答案后将 `resume_payload` 持久化并转 queued；新 Worker claim 后消费 payload，调用 `Command(resume={"answer": ...})`。payload 只能消费一次，使用状态 CAS 拒绝重复回答。

## 7. 事件、seq 与 SSE 迁移

当前 Fanout 先内存分配 seq、后批量落库，只能在单进程正确。目标是 `RunEventStore.append()` 成为唯一序号和持久化入口：

1. 在 PostgreSQL 内为每个 run 原子增加 `next_event_seq`，同事务插入 RunEvent。
2. 提交后通过 `EventNotifier.publish(run_id, seq)` 发送“有新事件”通知，通知不承载唯一事实。
3. SSE 始终按 `last_seq` 从 DB 读取；通知只负责立即唤醒查询，丢通知也可通过 heartbeat 补查。
4. token delta 仍是 ephemeral，允许丢失；阶段聚合事件、usage 和 run_done 必须持久化。

迁移期先实现 `PersistentEventSink + LocalEventNotifier`，再切 PostgreSQL NOTIFY 或 Redis，避免 Graph 感知传输后端。

## 8. 容量、计费和全局限制

### Run 级容量

分离“能否受理”和“能否立即执行”：每用户/global queued 上限防止队列滥用，global running 上限控制实际成本。超过 queued 配额才返回 429；仅执行槽已满时仍返回 202/queued。

### LLM 容量与速率

`LLMCapacityLimiter` 覆盖 Agent 模型、Router/Reflection 和 EvidenceExtractor 的全部调用。本地实现用进程级 Semaphore，独立 Worker 初期通过“Worker 数 × 每 Worker 槽位”保守配置，之后再引入 Redis token bucket 控制集群 RPM/TPM。Semaphore 只管在飞数，不替代速率与费用预算。

### Usage 与费用

在 RunExecutor 最外层安装 LangChain `UsageMetadataCallbackHandler`，收集全图实际 token；Agent middleware 记录 turn 粒度 usage；非 Agent 调用通过 tracing context 标记 node/operation。自定义 OpenAI-compatible provider 若不返回流式 usage，则使用 tokenizer 估算并显式标记 `estimated`，缺失不能当作 0。

持久化分为 `llm_usage_events` 调用明细和 Run 聚合。价格表按 provider/model/effective_at 版本化，保存计算时的 price version。预算包括单 Run model calls/input/output/cost/wall time，以及用户日/月与平台小时/日总额；达限时优先用已有 Evidence 降级交付，而不是简单报系统错误。

## 9. 工具缓存与可重放边界

工具缓存按 [恢复期工具缓存方案](恢复期工具缓存方案.md) 实施，通过 `ToolCache` 协议隔离存储。第一版使用 PostgreSQL，后续可换 Redis。`run_id/task_id/tool_call_id/operation_id` 只是观测关联，缓存键必须是规范化 query、URL/parser 版本或 `content_hash + research_direction + extractor/model/schema version`。

缓存解决 Worker 在 checkpoint 边界上重放已成功外部读的成本；它不替代 Run lease 和终态 CAS。取消、超时、未完整抽取或 5xx 不写正向缓存。

## 10. 配置与启动形态

新配置按职责分组：

```text
SERVICE_EMBEDDED_WORKER=true
RUN_MAX_GLOBAL_RUNNING=3
RUN_MAX_QUEUED_PER_USER=5
RUN_MAX_GLOBAL_QUEUED=100
RUN_LEASE_SECONDS=60
RUN_HEARTBEAT_SECONDS=20
LLM_MAX_IN_FLIGHT_PER_WORKER=4
FETCH_MAX_IN_FLIGHT_PER_WORKER=8
```

单进程模式由 API lifespan 创建 RunService 与 EmbeddedWorker。独立模式提供两个入口：

```text
python -m deepsearch_agent.service.api_server
python -m deepsearch_agent.worker
```

API 和 Worker 都只在自己的 lifespan 内创建/关闭 DB、checkpointer、HTTP/LLM 连接和 notifier，不在 Graph State 中传连接对象。

## 11. 分步迁移计划

### M0：锁定基线

**原因**：拆分控制面和执行面会触及取消、恢复、seq 和停机竞态，必须先将现有行为固化。

**工作**：保持现有 226 条测试；增加 RunExecutor 单元接缝；记录当前单进程真机的澄清、取消、停机恢复和 SSE 顺序基线。

**验收**：全量测试通过；Graph 文件无业务变更。

### M1：行为不变地提取 RunExecutor（已完成）

**原因**：先将 Graph 执行与 API 受理分开，才能在后续替换调度与队列，且这一步不需要 schema 迁移。

**工作**：新建 `RunExecutor`，搬迁 `_execute/_run_graph`、预览、终态持久化、事件排水。`RunManager` 暂时仍管理 `_tasks`、受理、取消、恢复和停机，通过窄接口调用 Executor。

**验收**：原有 RunManager/API 测试无需改业务断言；新测试确认 Manager 委托执行。

**实施结果**：Graph 构建与 `astream` 驱动、interrupt/终态持久化、事件预览及排水均已下沉到 `RunExecutor`；`RunManager` 不再包含 Graph 执行细节。

### M2：提取 RunService 并引入真正 queued（已完成）

**原因**：API 受理不应等价于立即消耗执行容量。

**工作**：`start()` 只插 queued 行；添加每用户/global queued 配额；新增 EmbeddedWorker 从 `RunQueue` 领取；完成/暂停时唤醒调度器。

**验收**：执行槽满时 POST 仍返回 202/queued；超过队列配额才 429；取消 queued Run 不启动 Graph。

**实施结果**：

- `RunService` 只负责配额判断并持久化 queued Run，不接触 Graph 或 `asyncio.Task`。
- `PostgresRunQueue` 以数据库中的 queued 行作为待办事实；`EmbeddedWorker` 按 `SERVICE_MAX_GLOBAL_RUNNING_RUNS` 填充执行槽。
- `SERVICE_MAX_GLOBAL_QUEUED_RUNS` 限制全局积压，`SERVICE_MAX_CONCURRENT_RUNS_PER_USER` 限制单用户未完成 Run；运行槽已满本身不再返回 429。
- queued Run 的取消先持久化用户意图，再处理可能已创建但尚未启动的协程，避免取消竞态遗留活跃状态。
- 启动恢复保留 queued 行并重新派发；服务停机仍将实际占槽任务登记为 interrupted。

**边界**：澄清答案的 resume payload 仍在 EmbeddedWorker 内存中，由 M4 持久化；事件 seq/fanout 仍是进程内事实，由 M5 外移。

### M3：Alembic + PostgreSQL claim/lease（已完成）

**原因**：没有数据库所有权记录，多 Worker 无法避免重复执行。

**工作**：引入迁移工具和 lease 字段；实现 `claim/renew/release/reap_expired`；终态写入校验 owner 和 attempt；增加两个并发 claimer 只有一个成功的 PG 集成测试。

**验收**：重复 claim 率为 0；Worker 丢租后不能写终态；过期 Run 能在 checkpoint 上恢复。

**实施结果**：

- Alembic `0001_initial → 0002_run_leases` 成为应用建库/升级路径；首次接管旧库时只 stamp 与旧模型完全一致的 0001，再执行追加式迁移。
- Run 新增 `lease_owner`、`lease_expires_at`、`attempt`；claim 通过单条 `UPDATE … RETURNING` 完成状态转换、owner 写入、attempt 递增和开始时间登记。
- PostgreSQL 候选选择使用 `FOR UPDATE SKIP LOCKED`；真实 PostgreSQL 双 claimer 集成测试确认同一 Run 只有一个 owner。
- Worker 按心跳续租；续租失败立即标记 lease lost 并取消本地 Graph。终态、失败和 awaiting_input 写入均校验 `run_id + owner + attempt + running`，旧 owner 无法覆盖新 attempt。
- 后台 reaper 持续回收过期 lease。存在 checkpoint 时以 resume work 重新 claim；没有 checkpoint 时明确失败并生成终结帧。正常停机通过 owner/attempt CAS 释放为 interrupted。

**边界**：claim 所有权已经支持多 Worker，但产品尚不能直接开启多进程：取消意图/resume payload 尚未完全持久化（M4），事件 seq 与实时 fanout 仍在进程内（M5），集群总并发/费用预算仍待 M6。

### M4：取消与 resume 持久化（已完成）

**原因**：API 进程不能再依赖持有 Worker 的 `asyncio.Task`。

**工作**：取消改为持久意图 + Worker probe；澄清答案改为持久 payload + 重新入队；通知先使用本地/PG，不把正确性寄托在 Pub/Sub。

**验收**：API 与 Worker 分开的测试中取消和澄清恢复均可用；重复回答被拒绝。

**实施结果**：

- Run 新增 `cancellation_requested_at` 与 `resume_payload`。取消 queued/awaiting_input/interrupted 时直接 CAS 到 cancelled；取消 running 时只持久化意图，owner Worker 在续租探测中观察后自我取消并以 lease CAS 写终态。
- 同进程的 `request_cancel()` 只是低延迟提示；跨进程正确性不依赖 API 持有 Worker 的 `asyncio.Task`。
- 澄清回答通过 `awaiting_input + resume_payload IS NULL → queued + payload` 单次 CAS 入队；重复提交被拒绝。claim 返回 payload，Worker 构造 `Command(resume=payload)`，payload 直到成功暂停/终结才清除，崩溃后仍可重放。
- 回归测试使用两个独立 RunManager 验证远端取消，并验证等待执行槽时答案已持久化且第二次回答失败。

### M5：持久事件序号与跨进程 SSE（已完成）

**原因**：进程内 seq/Fanout 是多 Worker 的最后硬阻塞之一。

**工作**：实现 `RunEventStore`、数据库原子 seq、`EventNotifier`；SSE 转为通知驱动的 DB tail；保留 token delta 的可丢失通道。

**验收**：Worker A 写事件、API B 的 SSE 立即收到；通知丢失后 heartbeat 仍能补齐；seq 无冲突。

**实施结果**：

- Run 新增 `event_seq`，旧事件 max seq 在迁移中回填。`RunEventStore.append()` 使用原子 `event_seq = event_seq + N RETURNING` 为批次分配互不重叠的连续区间，再与 RunEvent 同事务提交。
- Fanout 不再在进程内预分配权威 seq；引擎同步事件先进入无 seq pending，提交数据库后才通知本地订阅者。token delta 仍是明确可丢失的进程内旁路。
- PostgreSQL `LISTEN/NOTIFY` 唤醒其他 API 实例的 SSE；通知不携带业务帧。SSE 每次都按 `seq > last_seq` 从数据库 tail，因此通知重复或丢失都不影响正确性，并以 1 秒轮询兜底。
- 真实 PostgreSQL 集成测试同时验证两个 EventStore 并发编号为 1/2 且跨连接通知可达。

**边界**：到此事件回放与持久实时流已支持跨进程；token 级预览仍只在 Worker 与连接位于同一进程时可见。若独立 Worker 也需要逐 token 预览，应使用单独的可丢失通道，不把 token 写入 RunEvent 表。

### M6：Usage、成本和分层容量闸

**原因**：在没有实际 usage 时无法证明缓存收益，也无法安全扩大 Worker 数。

**工作**：接 LangChain usage callback；存明细/聚合/价格版本；实现 Run 槽、每 Worker LLM/fetch 槽、provider RPM/TPM 和 Run/user/platform 预算。

**验收**：所有模型调用都是 actual 或 estimated，不存在静默 0；预算耗尽能降级交付；并发不超配置。

### M7：恢复期工具缓存

**原因**：at-least-once 的节点恢复会重放 checkpoint 前已成功的外部读。

**工作**：按 L1 Search、L2 Fetch/Parse、L3 Evidence extraction 顺序接入 `ToolCache`；记录命中和节省 token/费用。

**验收**：在三类工具成功后、下一 checkpoint 前杀进程，恢复后外部调用计数不增加。

### M8：独立 Worker 与 Redis 可选升级

**原因**：在共享事实完成外移后，进程拆分才是启动方式变化，而不是业务重写。

**工作**：增加 Worker CLI、关闭 API embedded worker、运行 2+ Worker。只在 PG 轮询/通知、热缓存或集群限流成为实测瓶颈时，将对应协议后端替换为 Redis。

**验收**：多 Worker 无重复执行；任意 Worker 被 kill 后 Run 可接管；API 滚动发布不中断长任务。

## 12. 测试与发布策略

每个里程碑都使用绞杀式替换，新实现先隐藏在接口后，再切换生产路径。必须保留以下测试组：

- 状态机：创建、queued、claim、完成、失败、取消、awaiting_input、resume、lease expiry。
- 竞态：取消 vs 完成、shutdown vs mark-running、两 Worker claim、丢 lease vs 终态写。
- 恢复：无 checkpoint 判死、有 checkpoint 续跑、seq 续号、run_done 恰一次。
- 事件：DB replay + live tail 无洞、通知丢失可补、token delta 不落库。
- 容量：100 列表/详情请求、100 SSE、1/2/4/8 并发研究、provider 429/超时、预算耗尽。

上线顺序是单进程 shadow → EmbeddedWorker 生产 → API/Worker 同机分进程 → 两 Worker 灰度 → 水平扩容。每次切换都保留回退开关，且不在同一发布中同时替换队列、事件和缓存后端。

## 13. 完成定义

水平扩展改造完成需同时满足：

- API 不直接执行 Graph，Worker 不负责用户配额和受理。
- Run 队列、执行权、取消意图、resume payload、事件 seq、usage 和终态都有进程外的唯一事实。
- 一个 Run 任意时刻只有一个有效 lease owner，旧 owner 不能在丢租后写结果。
- 多 Worker 的事件可由任意 API 实例回放/实时输出，且 run_done 恰一次。
- Worker/API 可独立重启和滚动发布，不破坏 checkpoint 恢复。
- 通过实测 usage、费用、队列时间和并发指标确定扩容阈值，而不依赖估算用户数。
