# 服务化 P0 运行手册

把深度研究引擎包成多用户 web 服务的第一个里程碑：注册登录、提交问题、
实时看中文进度、收结构化报告、可取消、刷新可续看。

## 组成（全部单进程，无 Redis/队列/checkpointer——那些各有触发条件，见文末）

```
浏览器 (service/frontend/index.html)
   │  fetch + Bearer JWT；SSE 用 fetch 流式手解（EventSource 带不了 header）
   ▼
FastAPI (service/api.py) ── 鉴权每请求第一步 (auth.py, HS256 12h 无状态)
   ▼
RunManager (service/runs.py) ── asyncio.create_task 跑研究；runs/run_events 落库
   ▼
既有引擎零修改：build_graph(event_sink=CompositeSink(FanoutSink, JsonlSink))
事件 → FanoutSink 统一发 seq → projector 白名单投影 → SSE 帧(tick/status/error/done)
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

浏览器打开 `http://127.0.0.1:8080/` 注册即用；CLI `main.py` 不受影响。

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
- **进程边界**：Run claim、取消意图、resume payload、事件 seq 与 SSE DB tail 已持久化；聚合事件可跨 API/Worker 进程。token 级预览仍是进程内可丢失旁路，集群总容量/费用限制及独立 Worker 启动入口尚未完成，因此正式水平扩容仍按迁移计划的 M6–M8 推进。
- 越权与不存在同为 404；登录两错同为 401 文案。
- 事件流：`done` 帧恒最后且已落库（RunEvent），断线/刷新自动回放续接，seq 客户端去重。
- 配额：`SERVICE_MAX_CONCURRENT_RUNS_PER_USER`（活跃 queued+running 数，超限 429）。
- 每个 run 的原始事件仍写 `var/service/events/<run>.jsonl`（`SERVICE_JSONL_EVENTS=false` 关）。

## P0 明确不做（触发条件）

| 事项 | 何时做 |
|---|---|
| 服务端 token 吊销 / sessions 表 | 有真实安全需求或加"退出所有设备"时 |
| ~~checkpointer~~ **已接入（1/4 步）** | 状态快照已在 PG `checkpoints*` 表（thread_id=run_id，SQLite 部署自动跳过）；resume 驱动（reconcile 续跑+seq 续号）在 TODO P0②，做完才真正"杀而不死" |
| Redis（跨进程事件/配额）与任务队列 | web 与 worker 拆开或多机部署那天 |
| 登录限流、成本计量、Alembic、HTTPS/反代 | 对外部署前逐项补 |
| SSRF 守卫（fetcher 拦私网/元数据地址） | **任何公网托管前的一票否决项** |

## 开发门检

```bash
make check   # ruff + ruff format --check + pyright + pytest + coverage(≥70%)
# 无 Docker 过渡（非推荐形态）：SERVICE_DATABASE_URL=sqlite+aiosqlite:///var/service/dev.db
```
