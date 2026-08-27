# DeepSearch Agent TODO

> 当前阶段聚焦性能、运行可靠性和真实联网评估，不再累计已完成的架构重构事项。
>
> 状态：`[x]` 已完成、`[~]` 进行中、`[ ]` 待办。优先级：P0 必须先完成，P1 在主链稳定后完成，P2/P3 不阻塞交付。

## 当前基线（已完成）

- [x] 主流程：Router → Clarify → Supervisor → ResearchAgent → Writer（引用完整性校验） → Reflection → Render。
- [x] 已接入真实 Search / Fetch / HTML、PDF、DOCX、文本解析。
- [x] Evidence 具备 claim、quote、来源 URL、定位信息与确定性 quote 校验。
- [x] Writer 使用 `[[cite:evidence_id]]` 内部标记，本地统一渲染 `[来源N]` 和参考来源，且支持按需读取 Evidence。
- [x] Reflection 只输出问题严重级别；有 `fatal` 时才回到 Supervisor，成功后直接进入最终渲染。
- [x] 已有工具异常、HTTP 重试、事件、Trace、Writer 草稿审计和 90 条离线回归测试；Ruff/Pyright 已接入。

## P0：收敛领域模型与状态边界

### P0-A：统一核心数据契约（已完成）

- [x] 统一使用 Pydantic 领域模型：`Evidence`、`ReportBrief`、`Citation`、`ParagraphBinding`、`ResearchDirectionResult`、`WriterResult` 等已进入 State 通道。
- [x] `ResearchState` 只作为 LangGraph 顶层状态容器；字段直接引用领域模型，不再在节点间传递 `dict[str, object]`（`evidences`/`task_results`/`citations`/`paragraph_bindings`/`report_brief` 已模型化）。
- [x] State 已按 `run/research/writer/review` 分组，研究轮次与 Worker 状态语义已分离。
- [x] 删除语义错误字段：以 URL 充当内容的 `raw_documents` 改为 `source_refs`。
- [x] 为 Graph reducer 明确每个字段的覆盖、追加、不可变语义，避免节点重入时覆盖状态（`merge_evidences`/`merge_task_results`/`merge_unique` 三个语义 reducer 已落地，防御跨进程恢复的 dict→模型 coercion 集中在 reducer 内）。

验收：任何跨节点数据都能在运行时验证；核心流程不依赖裸字典字段名。

### P0-B：固定报告与引用边界（已完成）

- [x] 引用解析、绑定、编号重排和参考来源渲染位于 `reporting/`；Writer 只生成内部引用草稿。
- [x] Citation Verifier 只做确定性完整性校验，最终渲染由独立节点完成。
- [ ] 仅允许 `support=direct` 的 Evidence 支撑事实性结论；`partial/insufficient` 只能作为候选或限制说明。

验收：Graph 不解析 Markdown；报告渲染可脱离 LangGraph 独立测试。

### P0-C：拆分编排与业务实现（已完成）

- [x] `orchestration/nodes/` 已拆为 Router、Clarifier、Reflection、CitationVerifier 和 Render 薄适配层。
- [~] 将 Supervisor 拆为覆盖度决策、任务调度、URL 去重、事件发布四个协作对象（已迁移到工具调用协议：`ResearchDelegate` 派发 + `ResearchDecision` 充分性决策，URL 去重与审计保留在本地程序侧；协作对象拆分留待后续）。
- [x] ResearchAgent 已实现有界 Observe → Decide → Act 循环，可自主改写查询、选择来源、读取 Evidence、止损并返回方向结果。
- [x] 定义方向级结果契约：`TaskResult` 包含研究方向、已回答的子问题、结论/限定、Evidence、未解决缺口、失败分类、查询与来源轨迹和停止原因；成本字段待 P1 可观测性统一接入。
- [x] 明确 ResearchAgent 的自主边界：它只对已分配方向完成局部闭环，不决定全局研究是否足够、不撰写最终报告；全局覆盖与停止仍由 Supervisor 决定（ResearchAgent 已抽象为工具调用，结果以 ToolMessage 注入 Supervisor 上下文）。
- [x] Supervisor 负责全局研究决策；ResearchAgent 只负责方向闭环，Writer 不决定研究是否结束。
- [x] 已删除旧 `Finding`、旧 cite 标签、`plan` 字段、旧 Writer/Reader 适配层及不可达兼容分支。

验收：每个模块只因一种原因变化；核心文件不再同时处理流程、领域逻辑、渲染和日志。

## P0：运行可靠性与安全语义

### P0-D：资源、取消和错误分类

- [ ] 引入应用级 `ResearchApplication` / runtime container，统一拥有和关闭 `HttpClient`、事件 sink、trace recorder、LLM client。
- [ ] `build_graph()` 只接受已构造依赖，或显式返回可关闭的运行时对象；禁止库调用时泄露 HTTP 连接池。
- [ ] 建立统一错误分类：配置、可重试传输、来源不可用、解析、模型输出、质量拒绝、用户取消。
- [ ] 为每类错误定义重试策略、用户提示、事件码和是否继续其他任务；不要用宽泛 `except Exception` 隐藏语义。
- [ ] `CancelledError` 只传播和记录，不被重试或改写为普通失败。

验收：CLI、测试和嵌入式调用都能明确关闭资源；每次失败有稳定错误码和可读原因。

### P0-E：Evidence 安全闭环

- [ ] 移除“搜索标题 + 首段正文”自动升级为可写报告 Evidence 的降级逻辑；无 LLM 时只返回候选或明确研究未完成。
- [x] 段落可以通过 `ParagraphBinding.evidence_ids` 绑定多条 Evidence；当前不引入 ClaimLinker 或额外的 Claim 聚合层。
- [ ] 将 Evidence 质量规则写成确定性校验：quote 非空、来源一致、定位可复现、support 与可用范围一致。
- [ ] 为错误 quote、标题幻觉、条件遗漏、冲突来源、空正文补齐测试。

验收：任何最终事实性段落都能追溯到至少一条 `direct` Evidence，不能由搜索标题或模型常识伪装支撑。

> 设计决策：Evidence 保持最小、独立、可验证单元。多个来源共同支撑同一段落时，由 Writer 在引用标记中列出多个 `evidence_id`，由 Reflection 审阅支撑关系；只有出现明确的跨来源聚合或冲突检测需求时，才重新评估 Claim 层。

## P1：性能、恢复与可观测性

### P1-A：来源读取与缓存

- [x] 搜索内容回退：Tavily 请求 `raw_content`；网页读取失败时优先使用带 `retrieval_method=tavily_raw_content` 的回退正文，普通搜索摘要只能生成 `support=partial` Evidence。Writer 最低支持等级由 `AGENT_WRITER_MINIMUM_SUPPORT` 配置，默认 `direct`。
- [x] 长文分块抽取支持部分成功；单个 chunk 失败不会丢弃其他成功结果。
- [ ] SearchTool 增加 in-flight query 去重：并发 Worker 对相同 query 共享同一进行中 Task。
- [ ] 并发读取同一任务的候选 URL，设置来源级并发、超时和最大候选数；单个慢站点不阻塞整项研究。
- [ ] 引入网页快照、content hash、缓存 TTL 与来源质量评分；State 只保存快照引用。
- [ ] 增加 Writer 总输入预算：按研究主题相关度、support、confidence、来源去重选择 Evidence。

验收：长文偶发 API 失败不丢失整篇来源；重复查询/URL 不重复付费；研究延迟不被串行 Fetch 主导。

### P1-B：可观测性重整

- [x] 区分 `agent.log`、`events.jsonl` 和 `traces.jsonl`；摘要日志不再写入完整网页、Prompt、Evidence 或模型响应。
- [ ] 每个研究运行创建 `run_id` 目录，保存任务决策、草稿版本、Evidence、最终报告和失败原因。
- [ ] JSONL 事件增加轮转、保留策略和大小上限；审计正文按显式 debug/retain 开关保存。
- [ ] 记录每次 LLM 调用的阶段、模型、输入/输出 token、延迟、重试、错误类型；记录搜索调用、缓存命中、Fetch/解析/抽取结果。
- [ ] 保留可复现的 Writer 失败草稿，但默认不落 Prompt、原始网页正文或敏感内容。

验收：任一 run 可从 `run_id` 重建“为什么搜索、拿到什么 Evidence、为何写作/拒绝”；日志不会无界增长。

### P1-C：LLM 与工具策略

- [ ] 为 LLM 定义最小协议接口，不把 `ChatOpenAI`、`json_mode`、供应商 `extra_body` 参数散落在业务代码中。
- [ ] 让模型能力决定结构化输出模式；对不支持 JSON mode 的模型给出启动期诊断。
- [x] 将 `RetryPolicy` 纳入 `LLMConfig` / `Settings`，由应用装配一次并注入 `LLMInvoker`；`ainvoke_structured()` 不再隐式构造默认策略。
- [~] 将调用策略拆成两个互补层级：已分离传输重试（当前覆盖超时/断连）和 JSON/Pydantic repair，二者独立配置、独立预算；仍需补 429、可恢复 5xx、`Retry-After` 和阶段级事件。
- [ ] 统一 LLM 传输异常分类和退避；支持 429、服务端错误、网络错误与 `Retry-After`，并避免重试 `CancelledError`、认证/参数错误及不支持的 response format。
- [ ] 将结构化 repair 的次数、反馈最大长度、是否保留原 schema/消息、各调用阶段覆盖值收编至配置；普通自由文本调用不应无意义地走 JSON repair。
- [ ] HTTP 重试加入 jitter、`Retry-After`、可观测的 attempt 事件和按站点限速。
- [ ] 工具层保持确定性；LLM 只用于 query 设计、Evidence 抽取、Supervisor、Writer、Reflection。

验收：切换 OpenAI 兼容模型不会因硬编码请求参数静默失败；每次重试可解释且有边界。

## P1：测试、评估与工程卫生

### P1-D：自动化质量门槛

- [x] pytest、Ruff、Pyright、覆盖率与 pre-commit 已纳入项目配置。
- [ ] 建立 CI：lint → typecheck → unit → integration fixture → eval smoke。
- [ ] 增加 HTML/PDF/DOCX 解析 fixture；验证标题、段落、表格、编码、空壳页和验证码页。
- [ ] 增加端到端 fixture：Search → Fetch → Evidence → Writer → Reflection → Render，并覆盖失败、重试、取消。
- [ ] 将当前测试按 domain、tools、application、integration 分组，减少对私有函数和 monkeypatch 细节的耦合。

验收：主要重构由 CI 拦截；测试验证行为契约而非当前文件布局。

### P1-E：评估闭环

- [ ] 为 `eval/dataset.json` 实现执行器、结果存档和稳定 run ID。
- [ ] 建立小型真实 LLM smoke 集，单独控制预算与模型；不把网络/API 波动混入离线单元测试。
- [ ] 定义并记录：问题覆盖率、Evidence 支撑率、引用正确率、来源多样性、拒绝正确率、延迟、token、搜索调用数。
- [ ] 对每次 prompt 或流程改动运行固定 badcase 集；保留 baseline 和版本对比。
- [ ] 引入人工抽样复核，不把 LLM judge 当作唯一真值。

验收：能回答“此次改动是否提升研究质量、成本和稳定性”，而不只看到单次样例输出。

## 产品化路线：从研究引擎到完整应用

### P0：运行上下文与会话边界

- [ ] 建立明确的 `RunContext`：统一 `run_id`、`session_id`、取消信号、依赖容器和运行配置快照。
- [ ] 完善上下文管理：为 Supervisor、ResearchAgent、Writer 分别定义消息历史、工作集和摘要边界；按 token 预算压缩历史，而不是无限拼接消息。
- [ ] 将完整网页、Evidence 原文和报告草稿从 LLM 消息中分离；消息只携带目录、索引、摘要和工具结果。
- [ ] 修复入口仍传递旧 `raw_documents` 字段的问题，确保 CLI 与 `ResearchState` 契约完全一致。

### P1：持久化与恢复

- [ ] 接入 LangGraph checkpointer，实现同一 `thread_id` 的暂停、恢复、重试和取消后继续。
- [ ] 建立数据库模型：`runs`、`messages`、`tasks`、`sources`、`evidences`、`reports`、`reviews`、`events`；使用迁移工具管理 schema。
- [ ] 本地开发使用 SQLite，服务部署使用 PostgreSQL；数据库只保存元数据和可查询结果。
- [ ] 原始网页、PDF/DOCX 快照和大型 artifact 使用对象存储或文件存储，数据库保存 URI、hash、大小和生命周期信息。
- [ ] 为 Evidence、来源和搜索结果增加稳定版本、content hash、来源时间和缓存 TTL，支持重复研究复用。

### P1：上下文记忆策略

- [ ] 短期记忆（当前 Run/Thread 的消息、任务、Evidence 工作集）是首要需求，应先于长短期“记忆库”实现。
- [ ] 增加消息摘要、旧工具结果折叠和 Evidence 索引化；恢复时重新注入最新可用 Evidence，而不是盲目恢复全部正文。
- [ ] 长期记忆（用户偏好、历史研究结论、可复用知识）暂不作为主链必需项；只有明确的跨会话复用场景后，再设计权限、来源版本和失效策略。
- [ ] 长期记忆若落地，必须区分“用户偏好”“历史研究档案”和“可检索知识”，不能把三者混成一个向量库。

### P1：服务接口与 UI

- [ ] 提供 FastAPI 服务：创建研究、查询状态、获取事件、取消运行、恢复运行和下载报告。
- [ ] 使用 SSE 或 WebSocket 推送阶段、轮次、任务、来源读取和报告生成进度。
- [ ] UI 展示研究问题、方向任务、来源、Evidence、缺口、草稿、Reflection 意见和最终报告；支持查看引用回链。
- [ ] 将 UI 与 Agent 解耦，前端只消费稳定的 Run/Task/Event/Artifact API，不直接读取 LangGraph State。
- [ ] 增加认证、授权、配额、请求限流和用户级数据隔离。

### P1：部署与运营

- [ ] 提供可复现的开发/测试/生产配置，禁止把 API key、数据库连接和对象存储凭据写入代码或镜像。
- [ ] 增加 Docker/进程启动配置、健康检查、优雅关闭和后台任务恢复策略。
- [ ] 统一指标：运行成功率、阶段耗时、LLM token/费用、搜索/Fetch 命中率、Evidence 支撑率和队列长度。
- [ ] 为日志、事件、数据库和对象存储制定保留、删除、备份与隐私策略。

### P2：高级记忆与协作能力

- [ ] 在短期上下文稳定后，评估跨 Run 的语义检索与研究档案复用，而不是预先引入复杂记忆框架。
- [ ] 支持人工修订 Evidence、报告和任务方向，并保留版本与审计记录。
- [ ] 支持多用户、多项目、团队共享来源库和权限隔离。

## 文档维护规则

- [x] README 已描述当前真实拓扑、配置、运行方式、日志位置和已验证能力。
- [x] 文档已移除 Planner、Aggregator 和旧 Search/Reader Subagent 架构描述。
- [x] 文档维护规则要求状态机、数据契约和日志策略变更时同步更新。
- [x] 文档明确区分离线测试、真实 LLM smoke 和真实联网评估。
