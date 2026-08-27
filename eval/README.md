# Evaluation Dataset

这里保存 Deep Research Agent 的最小验证集。当前阶段只维护题目、类型和人工标注的关键要点，不自动调用外部 API，也不包含 Judge 实现。

每条样例包含 `id`、`query`、`type`、`difficulty`、`must_cover`、`requires_multi_round` 和 `source_expectations`。

当前先作为人工回归集使用：

```bash
uv run python main.py "<query>"
```

后续实现 `eval/run.py` 后，再统一运行全部样例并输出覆盖率、引用支撑率、耗时和调用次数。
