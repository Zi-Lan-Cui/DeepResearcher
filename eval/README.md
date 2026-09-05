# Evaluation Dataset

这里保存 Deep Research Agent 的最小验证集。当前阶段只维护题目、类型和人工标注的关键要点，不自动调用外部 API，也不包含 Judge 实现。

每条样例包含 `id`、`query`、`type`、`difficulty`、`must_cover`、`requires_multi_round` 和 `source_expectations`。

当前先作为人工回归集使用：启动 Web 服务后，逐条提交样例的 `query`，并按 `must_cover` 与 `source_expectations` 检查报告。

后续实现 `eval/run.py` 时通过 HTTP API 提交全部样例，不再绕过服务层直接调用 Graph，并统一输出覆盖率、引用支撑率、耗时和调用次数。
