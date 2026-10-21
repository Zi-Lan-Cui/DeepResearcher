# 评测

单轨设计：题目全部来自 DeepResearch Bench 中文 50 题，判分走官方 RACE 口径——
逐条 criterion 让被测文章与专家参考文章同台各打 0-10 分，按官方维度权重聚合后
以 target/(target+reference) 归一，0.5 = 与专家参考打平。正序与换序各判一次取
均值，消掉裁判的位次偏置。确定性门（引用完整性、run_done 恰一帧、事件 seq 连续）
对每份产物独立校验，作为通过前置，不产出质量分。

## 数据与版权

DRB 代码 MIT，数据集（题目/criterion/参考文章）另有许可。本仓库不存放 DRB 内容，
`drb.py` 运行时从本机 clone 的 `$DRB_ROOT` 解析；`split_drb_zh.json` 只含题号。
`evals/results/` 已被 gitignore，内容是我们自己的报告与分数。

## 准备

```bash
git clone https://github.com/Ayanami0730/deep_research_bench ../deep_research_bench
export DRB_ROOT=../deep_research_bench        # 或各命令显式传 --drb-root

uv run python -m evals.cli split --dev 20     # 生成固定切分，全组共用，改动会使 dev/holdout 归属漂移

export EVAL_BASE_URL=http://127.0.0.1:8080    # 被测服务：API + 至少 1 个 Worker
export EVAL_EMAIL=eval@local.dev EVAL_PASSWORD=***
export SERVICE_DATABASE_URL=postgresql+asyncpg://...   # 门禁判定需要 DB 直读 run_events
```

## 运行

DRB 中文题池共 50 题，切成两份 tracked 名单：`test30` 是实测集（三波各 10 题组齐，
对外口径只认它），`dev` 是开发迭代集（其余 20 题）。日常首轮用 dev 每题 1 次快速
定位问题，定稿阶段在 test30 上跑 pass@3 出成绩。

```bash
export EVAL_RESULTS_DIR=evals/results/dev-v1   # 每轮实验独立目录，新结果避免写进历史目录
uv run python -m evals.cli cases --profile dev
uv run python -m evals.cli run --profile dev --attempts 1
uv run python -m evals.cli score      # 确定性门，零费用
uv run python -m evals.cli judge      # RACE 双序判分（正序+换序），LLM 调用计入成本
uv run python -m evals.cli report     # 记分表 → summary.json
```

judge 每题每轮调两次：正序写 `judge_race.jsonl`，换序（参考文章放文章一）写
`judge_race_swapped.jsonl`，report 取两序 target/(target+reference) 的均值。
单题局部重评用 `run --ids`/`score --case`/`judge --case`；`run --attempts` 覆盖
每题次数，`--from-attempt` 续跑，`--concurrency` 批内并发。

## judge 配置

judge 身份取自评测侧环境变量：`EVAL_JUDGE_MODEL`（+ 可选 `EVAL_JUDGE_BASE_URL` /
`EVAL_JUDGE_API_KEY`，可整体指向另一提供商）。不配则回落到被测同一模型；同模型
评审存在 self-preference 偏高的已知风险，因此需要显式 `EVAL_SELF_JUDGE=1` 认领，
评审身份随之写入 `judge_meta.json`，report 的 `summary.judge` 原样带出——自评轮次
适合调试与冒烟，对外引用前建议换异构 judge。思考型网关需加大输出预算
（`EVAL_JUDGE_RACE_MAX_TOKENS=6144`），普通模型默认 2048 够用。

## 判分规则

- 判定发生在 criterion 级：同一调用内两篇文章各打 0-10 连续分，档位提示
  0-2/2-4/4-6/6-8/8-10；judge 逐条独立调用（整篇印象式打分容易被相邻条目带偏）。
- 单条调用或解析失败记双 None，聚合时排除出加权分母；整维无有效分按官方口径
  置零。缺分数量以 `missing_criterion_count` 上报，异常升高先查 judge 侧故障。
- case 通过：三门均通过 ∧ 双序均值 > 0.5（即稳定超过专家参考文章）。
- 过程指标（token/时长/缓存）随 artifact 记录，不参与通过判定。
- `pass@3`（三次至少一次过）与 `pass^3`（三次全过）并列报告；长任务服务对外
  建议以 `pass^k` 为主口径。

## 迭代纪律

提示词与阈值调整以 dev 20 题的失败归因为依据；holdout 30 题只取分数，不逐题
翻轨迹，以防针对题集调参。若向 DRB 官方排行榜提交，按其全量题协议另行执行。

## 对外指标

1. **Report Quality**：RACE 双序均值（分布、超参考比例）；
2. **Evidence Trustworthiness**：引用协议完整性门——正文每个 [证据N] 可溯源，
   来源条目 url/quote 非空；
3. **Engineering pass@3 / pass^3**：三门均通过 ∧ 双序均值 > 0.5 的轮次统计；
4. **Efficiency**：成功 run 的 token / 耗时，作成本趋势，不决定质量判定。

criterion 级分数与维度均值是上述指标的计算与归因细节，调试时可展开，
总览报表不平铺。
