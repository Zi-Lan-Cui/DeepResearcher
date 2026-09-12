# Supervisor 证据组织链路优化计划

## 1. 目标

在不推翻现有 Researcher、Supervisor 和 Writer 边界的前提下，打通：

```text
Researcher 方向结果
  → Evidence 工作集
  → ResearchAspect 跨方向论证组织
  → ResearchSynthesis 版本化综合
  → CoveredTopic 写作交接
  → Writer 按主题读取 Evidence
```

本轮只修复已有链路中的信息丢失、重复事实源和观察视图不一致，不增加新 Agent 或 Evidence 旁路。

## 2. 责任边界

### Researcher

- 完成一个可检索的局部方向；
- 产出通过逐字校验的 Evidence；
- 返回 `selected_evidence_ids + conclusion + remaining_gaps`；
- 不决定全局论证结构和文章章节。

### Supervisor

- 以 Evidence ID 管理活跃工作集；
- 将多个 Researcher 方向的 Evidence 组织成 `ResearchAspect`；
- 维护覆盖、冲突、缺口和下一步行动；
- 冻结最新 `ResearchSynthesis`，并交给 Writer。

`ResearchAspect` 是跨方向的证据支撑认知单元，不是 Researcher 任务方向，也不在概念上等于最终文章章节。

### Writer

- 消费 Supervisor 冻结的主题、综合和 Evidence 归属；
- 通过 `ReadEvidence` 按需读取完整 claim/quote/source；
- 负责文章表达和引用提交，不重做全局证据分类。

## 3. 推荐实施项

### P0：整理当前基线

1. 分组提交 Evidence `published_at` 与 locator 测试补全。
2. 分组提交 Researcher `answered_points` 移除与保守收尾。
3. 在基线干净后开始 Supervisor/Writer 改造。

### P1：明确 ResearchAspect 语义

- 更新 schema docstring、Supervisor prompt 和工具描述；
- 允许一个 Aspect 引用多个方向的 Evidence；
- 允许同一 Evidence 支持多个 Aspect；
- 禁止将 Aspect 描述为方向或固定章节。

### P2：统一 Supervisor EvidenceCard

建立单一的轻量观察函数，统一输出：

```text
evidence_id
claim
support
confidence
source_title
source_profile
published_at（非空时）
research_direction
```

默认不暴露 quote、audit_chunk、block 正文和 locator。

### P3：让 Aspect 成为 Evidence 选择的唯一事实源

- 从 `ReviseResearchSynthesis` 模型入参中移除 `selected_evidence_ids`；
- handler 按 Aspect 顺序和 Aspect 内 ID 顺序稳定推导并集；
- `ResearchSynthesis.selected_evidence_ids` 仍保留为程序生成的冻结结果；
- 返回尚未归入任何 Aspect 的活跃 Evidence ID，仅警告，不自动 Release。

### P4：强化综合稿校验

- CAS 校验 synthesis revision 和 working-set revision；
- `aspect_id` 全局唯一；
- Aspect 引用只能指向当前活跃 Evidence；
- 保留 covered/uncovered/partial/conflicted 的现有契约；
- 校验失败返回结构化 issues，不写入部分状态。

### P5：统一 ResearchSynthesis 观察视图

将 `_synthesis_snapshot()` 与 `_research_synthesis_observation()` 收敛到一个实现，完整保留：

```text
revision / based_on_working_set_revision
answer_goal / overall_summary
aspects[].aspect_id/topic/role/required/status/summary/evidence_ids/remaining_gap
selected_evidence_ids
open_gaps / conflicts / next_actions
readiness / decision_rationale
```

### P6：修复 ResearchAspect 到 Writer 的归属断点

- `CoveredTopic` 增加默认为空的 `evidence_ids`；
- `_report_brief_from_synthesis()` 保留 Aspect 的 Evidence 映射；
- `WriterDirective.evidence_ids` 由 `covered_topics[*].evidence_ids` 稳定推导；
- 保留旧 checkpoint 对缺失新字段的恢复能力。

### P7：Writer 按主题读取 Evidence

- 初始上下文呈现主题、角色、Supervisor 综合和建议 Evidence ID；
- 完整 Evidence 继续通过 `ReadEvidence` 批量按需读取；
- 同一 Evidence 即使属于多个主题，完整内容也只需读取一次；
- Writer 不得使用 Directive 允许集之外的 Evidence。

### P8：验证

- 单元测试覆盖跨方向组织、一证多 Aspect、非法 ID、死 ID、稳定并集、未归属警告、快照完整性和 checkpoint 兼容；
- 运行全量 pytest、Ruff 和 Pyright；
- 用现有 6 题小评测观察章节组织、引用错配、轮次和 token 变化。

## 4. 暂不实施

- `DirectionPoint`；
- Evidence/Aspect 细粒度 CRUD 工具；
- 独立 `ReportPlan` / `ReportSection`；
- Report Planner Agent；
- Supervisor 直接写报告；
- Supervisor 默认读取完整 quote；
- `EvidenceBinding(role, note)`；
- schema 大范围重命名；
- source digest、抽取诊断透传、截断统计和新 CoverageLedger。

## 5. 完成标准

1. `aspects[*].evidence_ids` 是 Evidence 归属的唯一模型事实源。
2. `ResearchSynthesis.selected_evidence_ids` 始终等于 Aspect Evidence 的稳定并集。
3. Supervisor 所有 EvidenceCard 视图一致，不暴露完整原文。
4. ResearchSynthesis 每轮观察不丢失 `role` 和 `required`。
5. Writer 能看到主题到 Evidence 的归属，不再重做全局材料分类。
6. 引用仍只能来自 `ReadEvidence` 返回的完整 Evidence。
7. 旧 checkpoint 可恢复，全量测试和静态检查通过。
