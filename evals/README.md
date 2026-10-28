# 评测

题目均来自 DeepResearch Bench 中文 50 题，判分按官方 RACE 规则——
逐条 criterion ，被测文章与专家参考文章对比各打 0-10 分，按官方维度权重聚合后
以 target/(target+reference) 归一，0.5 = 与专家参考相当。正序与换序各判一次取
均值，消除裁判位次偏置。

## 数据与版权

DRB 代码 MIT，数据集（题目/criterion/参考文章）另有许可。本仓库不存放 DRB 内容，
`drb.py` 运行时从本机 clone 的 `$DRB_ROOT` 解析；`split_drb_zh.json` 仅含题号。

## 准备

```bash
git clone https://github.com/Ayanami0730/deep_research_bench ../deep_research_bench
export DRB_ROOT=../deep_research_bench    

uv run python -m evals.cli split --dev 20     # 生成固定切分

export EVAL_BASE_URL=http://127.0.0.1:8080    # 被测服务：API + 至少 1 个 Worker
export EVAL_EMAIL=eval@local.dev EVAL_PASSWORD=***
export SERVICE_DATABASE_URL=postgresql+asyncpg://...   # 判定需要 DB 直读 run_events
```

## 运行

DRB 中文题池共 50 题，切成两份 tracked 名单：`test30` 是实测集，`dev` 是开发迭代集（其余 20 题）。

```bash
export EVAL_RESULTS_DIR=evals/results/dev-v1   # 每轮实验独立目录，新结果避免写进历史目录
uv run python -m evals.cli cases --profile dev
uv run python -m evals.cli run --profile dev --attempts 1
uv run python -m evals.cli score      # 确定性校验
uv run python -m evals.cli judge      # RACE 双序评分（正序+换序）
uv run python -m evals.cli report     # 记分表 → summary.json
```

judge 每题每轮调用两次：正序写 `judge_race.jsonl`，换序（参考文章放文章一）写
`judge_race_swapped.jsonl`，report 取正反两个顺序的 target/(target+reference) 分数均值。
单题局部重评分用 `run --ids`/`score --case`/`judge --case`；`run --attempts` 覆盖每题次数，可以使用`--from-attempt` 续跑，`--concurrency` 批内并发。

## judge 配置

judge 身份取自环境变量：`EVAL_JUDGE_MODEL`（+ 可选 `EVAL_JUDGE_BASE_URL` /
`EVAL_JUDGE_API_KEY`，可整体指向另一提供商）。若不写入，则使用被测同一模型；同模型评审存在 self-preference 偏高的已知问题，建议使用异构 judge评审。
评审身份写入 `judge_meta.json`，report 的 `summary.judge` 。思考型网关需加大输出预算（`EVAL_JUDGE_RACE_MAX_TOKENS=6144`），普通模型默认值即可满足需求。

## 判分规则

- 每个问题存在多个 criterion ，同一调用内 judge 对每篇文章的每个 criterion 各打 0-10 连续分，档位提示 0-2/2-4/4-6/6-8/8-10；judge 逐条独立调用。
- 单条调用或解析失败均记为 None，聚合时排除出加权分母；整维无有效分按官方规则
  置零。
- case 通过：确定性校验通过 AND 双序均值 > 0.5（即稳定超过专家参考文章）。
