# DeepSearch Agent

基于 LangGraph、LangChain 和 Pydantic 的深度研究原型：Router/Clarify 澄清问题，Supervisor 分配研究方向，ResearchAgent 自主检索和读取来源，Writer 生成草稿，Reflection 审阅，最后由本地程序完成引用校验与编号渲染。

## 流程

```text
Router → Clarify → Supervisor
                     ├─ ResearchAgent
                     │    ├─ SearchSources
                     │    ├─ ReadSources
                     │    ├─ Evidence 抽取与验证
                     │    └─ ResearchDirectionComplete
                     └─ Writer（本地引用完整性校验） → Reflection → Render
```

Supervisor 管全局研究，ResearchAgent 管单个方向，Writer 只负责成文。三者均采用“模型决策 → 工具调用 → 结果回到消息上下文”的 Agent loop。

### 环境

项目使用 uv 管理 Python 环境和锁文件：

```bash
uv sync
uv run python main.py "对比向量数据库的性能与成本"
```

Agent 会读取 `env/.env` 中的 LLM 配置，并通过 OpenAI 兼容接口调用模型。缺少必要配置时在应用装配期失败，不会执行一段无效研究。不要将 `env/.env` 提交到版本库。

统一配置入口为 `deepsearch_agent.config.get_settings()`。LLM、Agent 熔断参数和应用运行参数均由该模块读取并以 `Settings` 分发，业务模块不直接加载 dotenv。

基础可观测性位于 `deepsearch_agent/observability/`：文本日志用于快速排障，JSONL 事件用于机器分析，Trace 用于还原调用树。

默认运行时输出到 `var/logs/`（该目录已加入 `.gitignore`）：

- `agent.log`：关键摘要，包括节点耗时、搜索供应商和有效结果上限、任务结果、来源失败、Evidence 抽取耗时和最终状态；不写入完整网页、Prompt 或完整模型响应。
- `events.jsonl`：结构化生命周期事件，包括搜索、抓取、抽取、任务、研究轮次、Writer 和 Reflection 事件。
- `traces.jsonl`：Trace/Span 调用树。

每条 `events.jsonl` 记录都有 `event_id`、UTC `timestamp` 和可选的 `trace_id` / `span_id`。排障时应先按同一 `trace_id` 聚合：查看每轮派发了哪些任务、每个任务候选来源数、被拦截/失败的原因、实际抽取到多少 Evidence，以及最终引用校验为何拒绝报告。

本地日志/事件与 LangSmith 分工不同：本地文件用于离线审计、故障排查和后处理；LangSmith 用于完整的 LLM、Graph、Tool 调用链路追踪。当前尚未启用 LangSmith，后续可通过 LangChain/LangGraph 的 tracing 配置接入，两者可以并存。

`SearchClient` 和 `WebFetcher` 位于 `deepsearch_agent/tools/`。搜索通过适配层支持 Baidu、Tavily、SerpAPI；Fetch 使用 `curl_cffi` 并委托 `parsers/` 解析 HTML、纯文本、PDF、DOCX。SearchSources 只发现候选，只有 ReadSources 选择的来源才会抓取和抽取 Evidence。

Writer 使用内部 `[[cite:evidence_id]]` 标记，并在交给 Reflection 前完成正文绑定校验；渲染层按首次出现顺序生成 `[来源N]` 和参考来源表。研究不完整或节点失败时，最终报告会明确说明限制，不以模型常识补写事实。

如果当前机器的 uv 缓存目录不可写，可临时指定：

```bash
UV_CACHE_DIR=/tmp/deepsearch-agent-uv-cache uv sync
```

Python 版本由 `.python-version` 固定为 3.13；`pyproject.toml` 声明兼容 Python 3.12 及以上。

多用户服务默认将 API 与执行面分开：

```bash
docker compose up -d postgres
SERVICE_API_EMBEDDED_WORKER=false uv run python server.py
uv run python -m deepsearch_agent.worker
```

可启动多个 Worker，它们通过 PostgreSQL claim/lease 共享队列且不重复执行。本地单进程调试可设 `SERVICE_API_EMBEDDED_WORKER=true`。

运行回归测试和静态检查：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q -p pytest_asyncio.plugin
uv run ruff check src tests
uv run pyright
```

当前未完成事项见 [TODO.md](TODO.md)。

## 产品化状态

当前项目已经是可运行的研究引擎，但还不是完整的多用户应用。下一阶段优先补齐 Run/Thread 上下文管理、checkpointer、数据库与对象存储、服务 API、进度推送和 UI。短期记忆指当前研究运行的状态与消息历史，是恢复运行所必需的；长期记忆指跨会话的用户偏好或可复用知识，暂不作为主链依赖。
