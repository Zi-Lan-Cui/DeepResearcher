# 评测

双轨设计：报告质量走外部可比基准（DeepResearch Bench 中文题、官方 criterion 与
权重），系统行为走自研断言（澄清、口径冲突、证据不足时的保守交付、恢复与缓存）。
两套分数并列呈现、不合成总分——总分同时掩盖"文风好但引用差"和"工程稳但内容空"
两类失败，分开看才能归因。

## 结构

| 轨 | 题目来源 | 判分 | 用途 |
|---|---|---|---|
| 外轨 | DRB 中文 50 题（dev 20 / holdout 30，固定种子切分） | 专家二元 criterion × LLM judge（带参考文章对照）+ 确定性门 | 与外部系统可比 |
| 内轨 | `cases/behavior.jsonl` 15 题（B 澄清 / C 口径 / D 保守交付 / F 恢复缓存） | deterministic 断言为主，judge 只承担布尔断言 | 覆盖一次性"prompt→报告"类基准测不到的行为面 |

## 数据与版权

DRB 代码 MIT，数据集（题目/criterion/参考文章）另有许可。本仓库不存放 DRB 内容，
`drb.py` 运行时从本机 clone 的 `$DRB_ROOT` 解析；`split_drb_zh.json` 只含题号。
`evals/results/` 已被 gitignore，内容是我们自己的报告与分数。

## 准备

```bash
git clone https://github.com/Ayanami0730/deep_research_bench ../deep_research_bench
export DRB_ROOT=../deep_research_bench        # 或各命令显式传 --drb-root

uv run python -m evals.cli split --dev 20     # 生成固定切分，全组共用，改动会使 dev/holdout 归属漂移

export EVAL_BASE_URL=http://127.0.0.1:8080    # 被测服务：API + 至少 1 个 Worker（或 embedded）
export EVAL_EMAIL=eval@local.dev EVAL_PASSWORD=***
export SERVICE_DATABASE_URL=postgresql+asyncpg://...   # 行为断言需要 DB 特权读
```

## 运行

日常基线是 `core30`：20 道 dev 外轨题 + 10 道内轨题。首轮每题 1 次快速定位问题，
定稿前对关键题升到 3 次计稳定性指标。

```bash
export EVAL_RESULTS_DIR=evals/results/core30-v1   # 每轮实验独立目录，新结果避免写进历史目录
uv run python -m evals.cli cases --profile core30
uv run python -m evals.cli run --profile core30 --attempts 1
uv run python -m evals.cli score      # 确定性门 + 行为断言，零费用
uv run python -m evals.cli judge      # 二元 rubric，LLM 调用计入成本
uv run python -m evals.cli report     # 记分表 → summary.json
```

按轨/按功能运行：`run --track behavior`；`annotate` 导出 `results/annotate.csv`
人工填写，`annotate --import-csv` 回收并计算人机一致率与 Cohen κ。
Evidence 的 `audit_chunk` 存于 LangGraph checkpoint，runner 直读进 artifact 用于
核对 quote 与抽取上下文；该字段不进入 Agent 消息或 SSE 事件。

## judge 配置

judge 身份取自评测侧环境变量：`EVAL_JUDGE_MODEL`（+ 可选 `EVAL_JUDGE_BASE_URL` /
`EVAL_JUDGE_API_KEY`，可整体指向另一提供商）。不配则回落到被测同一模型；同模型
评审存在 self-preference 偏高的已知风险，因此需要显式 `EVAL_SELF_JUDGE=1` 认领，
评审身份随之写入 `judge_meta.json`，report 的 `summary.judge` 原样带出——自评轮次
适合调试与冒烟，对外引用前建议换异构 judge。当前引擎为 mimo，过渡期用
`EVAL_SELF_JUDGE=1`；换型只需设 `EVAL_JUDGE_MODEL`。

## 判分规则

- 判定发生在 criterion 级，取值 yes / no / unknown；judge 逐条独立调用（整篇
  印象式打分容易被相邻条目带偏）。
- 报告缺失 criterion 要求的内容记 no；unknown 只留给题目歧义、材料损坏或
  评审器故障。unknown 视为未通过，并计入 unknown rate 供 rubric 校准。
- 外轨 case 通过：确定性门全绿且四维加权 ≥ 75；内轨 case 通过：门全绿且
  全部行为断言为 yes。
- 过程指标（token/费用/时长/缓存命中）随 artifact 记录，不参与通过判定。
- `pass@3`（三次至少一次过）与 `pass^3`（三次全过）并列报告；长任务服务对外
  建议以 `pass^k` 为主口径。
- F1/F2（强杀/停机接管）需要 harness 管理 worker 进程，接线尚未完成，
  `--with-faults` 才会列出。

## 迭代纪律

提示词与阈值调整以 dev 20 题的失败归因为依据；holdout 30 题只取分数，不逐题
翻轨迹，以防针对题集调参。若向 DRB 官方排行榜提交，按其全量题协议另行执行。

## 对外指标

1. **Report Quality**：DRB 四维 criterion 加权质量分；
2. **Evidence Trustworthiness**：引用协议完整性 + claim–quote 支持度 + 来源质量
   的聚合——目前仅引用协议完整性层已自动化，其余两项补齐前，对外表述建议限定
   在已实现范围；
3. **System Reliability**：内轨用例 pass rate 与 `pass^k`；
4. **Efficiency**：成功 run 的 token / 耗时 / 费用，作成本趋势，不决定质量判定。

unknown rate 与人机一致率属于评审系统自身的健康度监控，不作为产品指标。
DRB 题内 criterion 与内轨状态机断言是上述指标的计算与归因细节，调试时可展开，
总览报表不平铺。
