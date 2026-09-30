# DeepResearcher

[![quality](https://github.com/Zi-Lan-Cui/deepresearcher/actions/workflows/quality.yml/badge.svg)](https://github.com/Zi-Lan-Cui/deepresearcher/actions/workflows/quality.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-%E2%89%A53.12-blue.svg)](pyproject.toml)

可自部署的深度研究服务:围绕一个复杂问题,自动完成澄清、检索、精读、证据整理、撰写与审阅,交付一份每条关键结论都能查到出处的研究报告。

## 为什么

多数"AI 深度研究"的产物是一段读起来可信、查起来无门的文字。这个项目把两件小事做实:**过程看得见**(此刻在检索什么、读哪个网站、攒了多少证据,实时推送),**结论查得回去**(报告用上标编号,文末逐条给出证据的标题、链接和支撑它的原文句子)。全部组件自托管:任务记录、证据和报告都留在你自己的数据库里,出网的只有你配置的模型与搜索调用。

## 特性

- 问题含糊时先澄清再开工,提交后关掉页面也没关系:排队、取消、断点续跑;
- 多个研究方向并行推进,互不依赖的检索同批发出;
- 证据必须逐字来自读过的原文,系统校验通过才算数;审阅不通过的稿件自动回流重写,证据不足则交付标注缺口的部分报告;
- 报告可一键复制,PDF 在浏览器本地生成;
- 多用户隔离,并发、token 与费用都有配额;
- 运行记录完整落库,网页、评测、回放读同一份数据;可选镜像到 Langfuse 看执行时间线。

## 快速开始

前置:Python ≥ 3.12、[uv](https://docs.astral.sh/uv/)、Docker。

```bash
git clone https://github.com/Zi-Lan-Cui/deepresearcher.git
cd deepresearcher
uv sync
cp env/.env.example env/.env
```

编辑 `env/.env`,填入模型三件套(`LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL_ID`,OpenAI 兼容接口均可)和至少一个搜索服务(百度 / Tavily / SerpAPI / 阿里云)。然后起依赖与两个进程:

```bash
docker compose up -d postgres redis
uv run python server.py                 # Web 服务,终端 1
uv run python -m deepresearcher.worker  # 研究执行进程,终端 2
```

打开 <http://127.0.0.1:8080> 注册提问。所有可调项在 [env/.env.example](env/.env.example) 逐项注释,包括研究轮数、各类预算和可选的流式进度开关。

## 使用示例

### 网页

注册登录后输入问题;若系统对研究边界有疑义会弹出一组选项,回答后开始研究;详情页实时滚动进度;完成后直接阅读报告、复制或导出 PDF。

### API

服务本身是一组普通 HTTP 接口,网页只是第一个客户端:

```bash
# 注册并拿到访问令牌
TOKEN=$(curl -s -X POST http://127.0.0.1:8080/api/register \
  -H 'Content-Type: application/json' \
  -d '{"email":"me@example.com","password":"<好密码>"}' | jq -r .token)

# 提出研究问题,返回任务 id(HTTP 202,后台执行)
curl -s -X POST http://127.0.0.1:8080/api/runs \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"query":"对比 Redis 与 MongoDB 在缓存场景下的取舍"}'

# 进度与结果:SSE 事件流(阶段、检索、证据计数……直到 done)
curl -N http://127.0.0.1:8080/api/runs/<run_id>/events \
  -H "Authorization: Bearer $TOKEN"
```

另有 `GET /api/runs` 历史列表、`POST /api/runs/{id}/resume` 回答澄清、`POST /api/runs/{id}/cancel` 取消。

## 项目结构

```
server.py                 开发入口(API 进程)
src/deepresearcher/
├── graph.py、nodes/、routing.py   LangGraph 流水线:路由→澄清→规划→写作→审阅
├── agents/               五个角色各自的工具、状态与子图(supervisor、researcher、writer…)
├── service/              对外一侧:API、worker、队列、持久化、事件流、前端
├── observability/        事件记录、OTel 引擎与运行日志
├── prompts/              各角色的系统提示词,外置成 Markdown
└── evidence/、reporting/、schemas/   证据模型、引用校验与渲染、数据契约
tests/                    单元与集成测试(make test)
evals/                    质量基准与行为断言两套评测(见 evals/README.md)
docs/                     技术报告与演进计划
docker/、docker-compose.yml         本地依赖与可选的自建观测栈
env/.env.example          全部配置项与注释
```

执行面只有一种形态:API 受理任务、worker 消费执行,两者只通过数据库交互,worker 可增减可重启。

## 开发

```bash
make quality    # lint、类型、测试、覆盖率(CI 每次推送跑同一套)
make verify-m8  # 真起 API 与 worker 双进程做集成验证,不产生模型费用
```

提交前跑 `make quality` 即可,没有额外的格式仪式;测试不依赖外部网络。

## 贡献

项目处于个人快速迭代期,暂无正式的贡献流程:遇到问题请开 issue,带上复现步骤和 `run_id` 更好;欢迎直接对文档和测试的改进。合并前以 `make quality` 全绿为门槛。

## 许可

[MIT](LICENSE) © 2026 Pan GuoDong
