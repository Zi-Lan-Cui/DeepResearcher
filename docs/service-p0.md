# 服务化 P0 运行手册

把深度研究引擎包成多用户 web 服务的第一个里程碑：注册登录、提交问题、
实时看中文进度、收结构化报告、可取消、刷新可续看。

## 组成（API / Worker 可独立水平扩展）

```
浏览器 (`service/frontend/index.html`)
   │  fetch + Bearer JWT；SSE 用 fetch 流式手解（EventSource 带不了 header）
   ▼
FastAPI 控制面 (`service/api.py` + `service/web/`)
   │  RunManager (`service/runs/`): 受理/取消/恢复，不执行 Graph
   ▼
PostgreSQL: Run 队列 / lease / checkpoint / RunEvent / usage / tool cache
   ▲
   │  claim + heartbeat + owner/attempt CAS
Worker 执行面 (`service/execution/`) × N
   │  build_graph(event_sink=CompositeSink(FanoutSink, JsonlSink))
   └─ 持久事件 → PostgreSQL NOTIFY 唤醒 API DB tail → `events/projector.py` 安全 SSE 帧
```

## 启动步骤

```bash
# 1. 数据库（唯一依赖 Docker 的部分）
docker compose up -d postgres

# 2. 配置（env/.env，参考 env/.env.example）
#    需要：LLM/搜索 key（引擎原有）+ 以下两项：
#    SERVICE_DATABASE_URL=postgresql+asyncpg://deepsearch:deepsearch@localhost:5432/deepsearch
#    SERVICE_JWT_SECRET=<32+字符>   生成: python -c "import secrets;print(secrets.token_urlsafe(32))"

# 3. 起服务（启动时自动执行 Alembic upgrade；生产发布也可先显式运行 alembic upgrade head）
uv run python server.py        # 默认 http://127.0.0.1:8080
```

默认分进程启动（共用同一 PostgreSQL）：

```bash
SERVICE_API_EMBEDDED_WORKER=false uv run python server.py
uv run python -m deepsearch_agent.worker   # 可启动多份
```

浏览器打开 `http://127.0.0.1:8080/` 注册即用。本地临时单进程模式可设 `SERVICE_API_EMBEDDED_WORKER=true`。Web/API 是唯一产品入口。

## curl 冒烟

```bash
TOKEN=** -s localhost:8080/api/register -H 'content-type: application/json' \
  -d '{"email":"a@b.c","password":"***"}' | python -c 'import sys,json;print(json.load(sys.stdin)["token"])')
curl -s localhost:8080/api/me -H "authorization: Bearer $TOKEN"
RUN=$(curl -s -X POST localhost:8080/api/runs -H "authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' -d '{"query":"什么是向量数据库"}' \
  | python -c 'import sys,json;print(json.load(sys.stdin)["run_id"])')
curl -N localhost:8080/api/runs/$RUN/events -H "authorization: Bearer $TOKEN"   # 看中文 tick 行直到 done
curl -s localhost:8080/api/runs/$RUN -H "authorization: Bearer $TOKEN" | python -m json.tool | head -20
```

## 语义速查

- 状态机：`queued → running → completed|failed|cancelled`；服务正常停机将活跃 run 记为非终态 `interrupted`（不写 `run_done`）。重启时**分诊** `queued|running|interrupted`：有 checkpoint 的孤儿 run 复活续跑（SSE 播 `resuming`，seq 从库中 max 续号，真机 kill -9 验收通过），无 checkpoint 的才判 `server_restart` 并补写终帧。
- **进程边界**：API 默认不执行 Graph，独立 Worker 自主轮询持久队列。Run claim、取消意图、resume payload、事件 seq、SSE DB tail、usage/预算、全局 Run 槽和三层工具缓存由 PostgreSQL 协调。token 级预览仍是 Worker 本地可丢失旁路，LLM RPM/TPM、search/fetch 容量和同键 single-flight 是每 Worker 限制。
- 越权与不存在同为 404；登录两错同为 401 文案。
- 事件流：`done` 帧恒最后且已落库（RunEvent），断线/刷新自动回放续接，seq 客户端去重。
- 配额：`SERVICE_MAX_CONCURRENT_RUNS_PER_USER`（活跃 queued+running 数，超限 429）。
- 每个 run 的原始事件仍写 `var/service/events/<run>.jsonl`（`SERVICE_JSONL_EVENTS=false` 关）。

## P0 明确不做（触发条件）

| 事项 | 何时做 |
|---|---|
| 服务端 token 吊销 / sessions 表 | 有真实安全需求或加"退出所有设备"时 |
| ~~checkpointer~~ **已接入（1/4 步）** | 状态快照已在 PG `checkpoints*` 表（thread_id=run_id，SQLite 部署自动跳过）；resume 驱动（reconcile 续跑+seq 续号）在 TODO P0②，做完才真正"杀而不死" |
| Redis（跨进程热缓存/限流） | PG 轮询、冷键惊群或跨 Worker 限流成为实测瓶颈时 |
| 登录限流、HTTPS/反代 | 对外部署前补；成本计量和 Alembic 已完成 |
| SSRF 守卫（fetcher 拦私网/元数据地址） | **任何公网托管前的一票否决项** |

## 开发门检

```bash
make check   # ruff + ruff format --check + pyright + pytest + coverage(≥70%)
make verify-m8  # 真实多进程 + PostgreSQL，但不调用 LLM/搜索/抓取
# 无 Docker 过渡（仅支持单进程）：SERVICE_DATABASE_URL=sqlite+aiosqlite:///var/service/dev.db
```

`verify-m8` 会在随机本地端口启动真实 Uvicorn API 和两个 Worker 子进程，但 Worker 使用测试专用的 checkpointed Graph。默认验证：

- 100 个同时 SSE 连接和 100 个列表/详情请求；
- 1、2、4、8 并发 Run 都只进入一个 Worker；
- 运行中销毁并重建 API，Run 仍由 Worker 继续持有；
- 对 lease owner 发 `SIGKILL`，第二个 Worker 在 lease 过期后以 `attempt=2` 从 checkpoint 接管；
- Run 最终 completed，event seq 无洞，`run_done` 恰好一次。

失败时脚本保留 `/tmp/deepsearch-m8-*` 中的子进程日志；成功时自动清理。这只验证服务调度容量，不代表真实 LLM/provider 吞吐；付费链路压测仍需显式单独运行。
