# DeepResearcher

DeepResearcher 是一个可本地部署的深度研究服务。它会围绕用户的复杂问题自动检索、阅读和组织公开资料，生成附带可追溯引用的研究报告。

## 主要功能

- 在问题范围不明确时向用户发起澄清，并根据回答继续研究。
- 自动完成多轮搜索、网页阅读、资料比较和证据整理。
- 为关键结论绑定原文证据，在报告末尾生成参考来源。
- 实时展示研究进度，支持任务排队、取消、中断恢复和历史查看。
- 支持多用户 Web 使用，报告可一键复制或下载为 PDF。

## 快速开始

### 1. 准备环境

需要 Python 3.12+、[uv](https://docs.astral.sh/uv/) 和 Docker。

~~~bash
uv sync
cp env/.env.example env/.env
~~~

编辑 env/.env，至少配置：

- LLM_API_KEY
- LLM_BASE_URL
- LLM_MODEL_ID
- 一个搜索服务：百度、Tavily、SerpAPI，或使用默认凭据链的阿里云 DTS AI

### 2. 启动依赖服务

~~~bash
docker compose up -d postgres redis
~~~

### 3. 启动 DeepResearcher

打开两个终端，分别运行：

~~~bash
SERVICE_REDIS_PREVIEW_ENABLED=true uv run python server.py
~~~

~~~bash
SERVICE_REDIS_PREVIEW_ENABLED=true uv run python -m deepresearcher.worker
~~~

默认访问地址：<http://127.0.0.1:8080>

## 使用方法

1. 在网页中注册并登录。
2. 输入需要深入研究的问题，点击“开始研究”。
3. 如果系统需要确认研究范围，选择选项或输入补充说明。
4. 在任务详情页查看实时进度。离开页面不会终止任务。
5. 研究完成后阅读报告与参考来源，或使用“复制”和“下载 PDF”导出结果。

## 配置

所有可配置项及默认值见 [env/.env.example](env/.env.example)。常用配置包括：

- 模型、搜索服务和超时时间；
- 单次研究轮数与证据数量；
- 用户和服务的并发上限；
- 数据库、Redis 与登录令牌；
- Token 和费用上限。

## 容量假设

- 事件 Hub 是进程内的:`_OPEN_MAX=2048` 个 open run 登记,超额按插入序回收最早者,
  未落库的 pending 会丢弃并记 `run_event_pending_dropped_on_evict` warning(已落库的
  不受影响)。前提是单进程(API 或 worker)并发 run 数远小于该值。
- `run_done` 幂等去重只覆盖同进程;API 与 worker 各持一个 hub,同一 run 理论上可能
  落库两条 done,SSE 首条 done 即终止,客户端无感。
- `ProviderRateLimiter` 的 RPM/TPM 窗口是 per-worker 的,按估算 token 记账、不与实际
  返回对账:多 worker 时有效限额约为配置值 x N。单 worker 部署不受影响;多 worker 请
  按 worker 数折算配置,或替换为共享计数实现。
- 搜索查询结果缓存(`SearchTool._query_cache`)随 run 结束整体回收,不设进程级上限。

## 公网部署

默认配置只监听本机地址。如果需要公网访问，请在应用前配置 HTTPS 反向代理，并使用强随机 SERVICE_JWT_SECRET；不要将开发配置直接暴露到公网。
