# Phase 1：受控异常信号与新研究协议

日期：2026-11-19

## 1. 研究问题

Phase 1 不继续用真实测试事件调 Coverage 参数。它回答三个可证伪的问题：

1. temporal-soft 是否只在持续漂移类信号上有效，而对孤立点异常没有独立收益；
2. 点级残差、时序上下文和二者组合是否响应不同类型的异常信号；
3. 在相同受控事件上，instance-soft/Coverage 模型是否比官方对照更稳，iForest
   是否只在单点高幅度异常上占优。

本阶段的数据不是新的外部数据集。它以 CERT r4.2 和 CLUE-LDS 的真实正常行为
为母体，构造固定 seed、固定规则、有精确起止时间的半合成事件。它可以支持
机制和边界结论，不能支持外部有效性声明。

执行状态：2026-11-19 已完成 3-seed 矩阵。CERT 两个 profile 各使用 96 个
背景用户；CLUE 在“训练和评价期各至少 20 个活跃日”的固定筛选下只有 54 个
合格用户，事件用户仍为 24 个。该实际规模写入 manifest，不做事后放宽。

## 2. 数据构造

### 2.1 底座

- `r42-temporal`：CERT r4.2，比较 `T3_instance_soft` 与 `T4_full_soft`；
- `r42-coverage`：CERT r4.2，比较 `A0-official` 与 `A1-Cov`；
- `clue-coverage`：CLUE-LDS，比较 `A0-official` 与 `A1-Cov`。

只保留评价期前 64 天真实历史作为因果上下文。训练期末尾 64 天用于分数校准。
标签 seed 与模型 seed 分离；任何模型都没有在注入事件上训练或选参。

### 2.2 事件类型

| 名称 | 持续时间 | 构造 |
| --- | ---: | --- |
| `point` | 1 天 | 在最高方差通道上注入单日幅度尖峰 |
| `sustained` | 7 天 | 注入持续多通道偏移 |
| `drift` | 10 天 | 从低幅度线性增加到目标幅度的缓慢漂移 |
| `switch` | 7 天 | 将目标用户窗口按比例替换成另一用户的行为模式，模拟账号切换 |

每个强度独立生成数据。默认强度为 `0.5/1.0/2.0`。每个底座默认选择 128 个
背景用户，其中 32 个用户每人接收一个事件。所有方法使用同一目标用户、同一
事件起点和同一强度。

## 3. 异常信号

对每个模型计算 `mask_last` 与无遮挡表示的 L1 残差，并用训练期鲁棒中位数和
MAD 校准：

```text
point    = max(0, (raw_residual - median) / MAD_scale)
context  = 5-day causal mean of point
combined = 0.5 * point + 0.5 * context
```

上下文只使用当前及过去分数，不读取未来。iForest 使用与模型完全相同的训练
尾段、原始特征、目标用户、事件和评价协议。

## 4. 评价

- 点级：PR-AUC、ROC-AUC；
- 事件级：每天固定 K=5/10，用户冷却期 7 天；
- 延迟：事件起点到首次命中的天数；
- 主对照：同一事件和强度下的配对差值；
- 不做事后调参，不按结果删除失败强度或失败事件类型。

## 5. 解释规则

支持 temporal-soft 桥接的最低形式是：

- `T4-T3` 在持续漂移上方向为正；
- 该收益不能同时由点尖峰的普遍变差解释；
- 结果必须在多个 seed 上同向。

如果只有时间平滑几何改善，但 PR-AUC 和事件召回没有提高，只能写“局部
平滑效应存在，端到端收益未确认”。

如果受控数据上 Coverage 优于对照，而真实 CLUE 仍为负，论文必须并列报告，
并把差异归因于事件构造、先验信息量或样本功效，不能覆盖真实负结果。

## 6. 运行

```powershell
python research/ccf-b/phase1/controlled_signal_study.py `
  --profile r42-temporal `
  --seeds 20261116 `
  --output-dir "<path-to-cert-runtime>\external\phase1_r42_temporal_pilot"
```

```powershell
python research/ccf-b/phase1/controlled_signal_study.py `
  --profile clue-coverage `
  --seeds 20261116 `
  --background-users 96 `
  --event-users 24 `
  --output-dir "<path-to-cert-runtime>\external\phase1_clue_coverage_pilot"
```

每个输出目录必须包含：

```text
manifest.json
metrics_by_seed.csv
summary.csv
paired_comparisons.csv
events.csv
result.json
```

本次结果见 `PHASE1_RESULTS_20261119.md`。最重要的冻结事实是：受控数据上的
A1-Cov 正向与真实 CLUE 的负结果相反，因此后续论文不得把受控结果写成覆盖率
外部复现。

## 7. 禁止声明

- 不得称为新的真实 CERT 或 CLUE 数据；
- 不得把受控结果写成外部验证；
- 不得根据测试结果修改事件类型或强度后重报；
- 不得用受控数据上的正结果覆盖 CLUE 上 Coverage 的负结果。
