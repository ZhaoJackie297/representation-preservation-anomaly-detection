# CCF-B 计划勘误 v1.2.7

日期：2026-11-18

本勘误回应 v1.2.6 审阅，冻结以下内容：

1. CLUE-LDS 历史数字与 canonical 数字的对账；
2. 状态分类，避免把机制不可识别、功效不足和数据门禁失败混写；
3. CERT r4.2/r5.2 画像先验与 kNN 替代带宽审计；
4. 批次携带关系信息的控制变量；
5. 覆盖步数、A1+/A9 调度和 EMA 距离边界；
6. r5.2 Gate 的功效代理结果；
7. Pinsker 型假负样本偏差界。

本勘误写在任何新的 A3/A4 性能结果之前。只允许通过新的、同样在看结果前冻结的勘误修改。

面向执行的整合路线见 `CCF_B_UPGRADE_PLAN_v1.md`。

SoftCLT 实例损失的权重尺度审计见 `SOFTCLT_WEIGHT_AUDIT.md`；在归一化/
质量通道定义冻结前，不运行新的关系通道归因实验。

## 1. CLUE-LDS 数字对账

### 1.1 历史记录

`PLAN_ERRATA_v1.2.5.md` 和 `CLUE_LDS_PHASE0_RESULTS.md` 的早期版本记录了开发阶段的：

```text
label_seed = 20261116
pairs = 20
events = 40
relation_graph isolated_users = 568 / 838
```

这些数字属于 v1.2.5 开发历史，不是当前确认数据。历史文档不删除，避免改写审计轨迹。

### 1.2 当前 canonical

v1.2.6 起冻结：

```text
label_seed = 20261101
pairs = 24
events = 48
relation_graph isolated_users = 565 / 838
```

标签文件：

```text
hijack_labels_24pairs_20261101.csv
SHA-256 = 2441744DED393381C9B42D3E1EFB53B11137810D8F94611FCF76BC0C67D248CA
```

逐文件哈希：

```text
research/ccf-b/clue_lds/artifact_hashes_20261118.json
```

当前 canonical 的 24 pairs、48 个用户级事件和 565 个孤立用户必须同时出现。任何使用 20/40/568 的数字进入训练或统计表的 run 标记为 `STALE_CLUE_DEVELOPMENT_DATA`，不得与 canonical 结果合并。

### 1.3 对 v1.2.5 的处理

v1.2.5 保留为“开发记录”，不修改其历史内容；v1.2.6/v1.2.7 的 canonical 覆盖它以
及所有后续主文档。引用时必须写明版本，不能用旧 manifest 覆盖 canonical manifest。

## 2. 状态分类

### 2.1 固定状态码

| 状态码 | 含义 | 禁止表述 |
| --- | --- | --- |
| `IMPLEMENTATION_BLOCKED` | 采样器、覆盖、形状或训练接线未通过 | 不能写成方法失败 |
| `DATA_GATE_FAILED` | 数据许可、时间、标签、用户映射或切分不满足 | 不能写成方法失败 |
| `METHOD_NOT_SUPPORTED` | 机制在当前先验/数据上不可识别或没有支持证据 | 不能写成数据不可用 |
| `UNDERPOWERED` | 当前事件簇和 MDE 无法可靠区分目标效应 | 不能写成方法无效 |
| `PENDING` | 尚未执行对应门禁或矩阵 | 不能写成通过 |

### 2.2 当前状态

| 对象 | 当前状态 | 理由 |
| --- | --- | --- |
| CLUE-LDS 数据/标签/图/覆盖 | 数据门禁通过 | 24 pairs、48 事件和 10-seed 覆盖已通过 |
| CLUE-LDS 外部效应确认 | `UNDERPOWERED` | 24 个 pair 的召回 MDE 约 23.0–29.5 pp |
| CLUE-LDS 画像锚机制 | `METHOD_NOT_SUPPORTED` | 主带宽和 kNN 带宽均触发 `FLAT_PROFILE_PRIOR` |
| CERT r4.2 画像锚机制 | `METHOD_NOT_SUPPORTED` | m=5/10、主带宽和 kNN 带宽均平坦 |
| CERT r5.2 画像锚机制 | `METHOD_NOT_SUPPORTED` | m=5/10、主带宽和 kNN 带宽均平坦 |
| r5.2 严格双终点 Gate | `UNDERPOWERED` | 99 事件的代理 MDE 不能支持过严的双终点主张 |
| A0/A1 训练入口接线 | smoke 通过 | 真实 CLUE 数据 2-step smoke 通过，正式矩阵未跑 |
| LANL | `PENDING` | 尚未通过时间、事件簇和图适配门禁 |

“平坦先验”是 `METHOD_NOT_SUPPORTED`，不是 `DATA_GATE_FAILED`；“CLUE 低功效”是
`UNDERPOWERED`，不是“方法失败”。两者都不能被数据门禁状态掩盖。

## 3. CERT 画像先验审计

### 3.1 冻结的替代带宽

主带宽仍为全体 pairwise 训练距离中位数。预注册一个不按性能选择的替代带宽：

```text
h_knn = median_u d(u, NN_k(u))
k = 10
```

这不改变主结果；只用于检验“主核过宽导致平坦”这一实施假设。

### 3.2 结果

r4.2（1,000 用户，47 维画像）：

| 先验 | mean H10 | fraction > 0.95 log 10 | 状态 |
| --- | ---: | ---: | --- |
| 主带宽 | 2.30154 | 1.000 | `FLAT_PROFILE_PRIOR` |
| kNN 带宽 | 2.22480 | 0.889 | `FLAT_PROFILE_PRIOR` |

m=5 时两者也均触发平坦规则。r5.2（2,000 用户）得到同方向结论：

| 先验 | mean H10 | fraction > 0.95 log 10 | 状态 |
| --- | ---: | ---: | --- |
| 主带宽 | 2.29736 | 0.9975 | `FLAT_PROFILE_PRIOR` |
| kNN 带宽 | 2.24142 | 0.8905 | `FLAT_PROFILE_PRIOR` |

详细 JSON：

```text
research/ccf-b/cert_profile_information_r42.json
research/ccf-b/cert_profile_information_r42_m5.json
research/ccf-b/cert_profile_information_r52.json
research/ccf-b/cert_profile_information_r52_m5.json
```

### 3.3 后果

- CERT 和 CLUE-LDS 都不能把 A3/A5b 的画像锚贡献作为已支持的机制结论；
- A3/A4 的端到端性能仍可测，但增益只能记为“支撑集/表示反馈/批次共现等待分解机制”；
- 不得写“A5b 证明画像无用”；
- 不得把平坦先验当成数据门禁失败。

## 4. 批次选择与分解控制

### 4.1 新增控制

共同覆盖批次表仍由 A3/A5a/A5b 联合构造，因此批次本身携带部分画像关系信息。新增：

| 变体 | 批次表 | 软权重 | 目的 |
| --- | --- | --- | --- |
| `A0-A3batch` | 只由 A3 边选择的共同覆盖表 | 硬负样本 | 测量批次本身携带的 A3 邻域信息 |
| `A1-Cov-A5a` | 由 A5a 边选择的共同覆盖表 | 固定画像软权重 | 分离“软权重”与“A3 批次选择” |
| `A1-Cov-uniform` | 与 A1-Cov 同一批次表 | 支撑集内均匀权重 | 分离“关系权重形状”与“软权重存在” |

分解顺序改为：

```text
A0-official
  -> A0-matched
  -> A0-A3batch
  -> A1-Cov-uniform / A1-Cov-A5a
  -> A1-Cov
  -> A3/A4
```

任何 `A1-Cov > A0-matched` 的差值同时报告 `A0-A3batch` 对照；若 A0-A3batch
已取得同量级收益，不能把差值归因于画像软权重。

`A1-sparse vs A1-sparse-EMA` 必须在同一张主表中报告，用于量化“支撑集是否加入表示信息”的贡献。

### 4.2 覆盖步数口径

覆盖门禁是对**实际进入训练的批次表**生效的，不是对离线边矩阵生效。冻结：

```text
B_stage = max(B_base, required_coverage_batches)
```

- A0-matched、A1-Cov、A1-Cov-A5a、A1-Cov-uniform 使用完整 `B_stage`；
- A0-official 使用原版随机 shuffle 的 `B_base`；
- A1+ 每阶段重新生成同 seed、同 stage 的覆盖表，不能只重放第一张表；
- A3/A4 每个新图阶段重新生成目标边和覆盖表；
- A9 的三段各自重新生成覆盖表，并逐段重置优化器；
- 若实际训练步数少于该阶段覆盖表长度，run 直接标 `INVALID_COVERAGE_BUDGET`。

### 4.3 自权重与邻居质量

主模型仍为自权重 `1`、邻居质量 `rho=1`。新增预注册敏感性：

```text
rho in {0.25, 0.5, 1.0, 2.0}
```

必须报告每个锚点的自权重/邻居质量比、有效正样本数和实际进入 loss 的 pair mass。
不能用“自权重 1”暗示邻居质量已经被验证为最优。

## 5. EMA 与 beta 数值规则

### 5.1 距离边界

表示距离仍以行中位数归一化，但增加固定上界：

```text
d_tilde(u,v) = min(d_raw(u,v) / max(row_median(u), 1e-6), 10)
```

若任一阶段 `d_tilde` 触发上界的比例超过 `5%`，该阶段标
`EMA_DISTANCE_SATURATED`；只能报告端到端结果，不能声称 EMA 机制有效。

### 5.2 教师质量门禁

在 A3 进入第二阶段前，教师表示必须满足：

```text
NCA@10_teacher >= 0.95 * NCA@10_A1plus
ER_teacher / ER_A1plus >= 0.90
```

否则：

```text
EMA_TEACHER_NOT_QUALIFIED
```

不允许把不合格教师反馈当作 A3 机制证据。

### 5.3 beta 和支撑集

- `beta=0` 作为独立端点处理，不用二分搜索近似；
- 二分下限和容差必须记录；
- `P_m` 重归一化在同一个支撑集上进行；
- 不满足 KL 约束时标记 `BETA_SOLVE_FAILED`，不得静默回退 `beta=1` 后继续主比较；
- `beta`、实际 KL、熵和 clip fraction统一记录。

## 6. r5.2 Gate 功效代理

使用历史 r4.2 T3/T4 三种子和 r5.2 T4 三种子的 per-seed 结果做代理，不把它当作
A4 对 A1+ 的正式功效分析：

- r4.2 的 K10 基线召回约 47.2%，24 个事件；独立事件近似 MDE 约 40.4 pp，
  `rho=0.5` 近似约 28.5 pp；
- r5.2 的 K10 基线召回约 17.2%，99 个事件；独立事件近似 MDE 约 15.0 pp，
  `rho=0.5` 近似约 10.6 pp，`rho=0.8` 近似约 6.7 pp；
- r5.2 的 T4 bank Euclidean 对 T4 original 的历史差值均值约 6.7 pp，不能证明
  A4 对 A1+/A9 的严格双终点 Gate 有足够功效。

详细结果：

```text
research/ccf-b/cert_gate_power_proxy_20261118.json
```

因此：

1. 当前严格 Gate 标为 `UNDERPOWERED`，不得把未过门禁自动写成方法失败；
2. 正式 Gate 必须先做事件/用户簇层面功效模拟；
3. 在正式功效结果出来前，不承诺“两个终点均优效”可由 99 个事件可靠检出。

## 7. 非劣界修正

旧界：

```text
PR: -0.0005
K10: -max(0.02, 2/N)
```

在 r5.2 上 PR 界约为基线的 7%，K10 界约等于两个事件，偏宽。v1.2.7 预注册更严格的
临时界：

```text
PR_margin = -min(0.0005, 0.05 * PR_A1plus)
K10_margin = -max(0.01, 1 / N_events)
```

含义：

- PR 不能同时放宽到 0.0005 和 5% 的较大者；只能取更严格者；
- K10 默认只允许一个事件步长，并用 1% 绝对下限保护极小事件集；
- 这些是 Gate 的临时上界，最终 `D_ext` 仍需按事件簇 MDE 复核；
- 任何改变只能通过看结果前冻结的新勘误。

## 8. Pinsker 型假负偏差界

设真实同类关系行分布为 `W*_u`，模型软图为 `W_u`，均在同一用户支撑集上归一化。
若

```text
KL(W_u || W*_u) <= delta
```

且用户特征向量长度不超过 `B`，则加权重心满足

```text
|| c_W(u) - c_W*(u) ||_2 <= C * sqrt(delta / 2)
```

其中 `C` 由特征空间直径决定，可以用 `2B` 或经过验证的紧常数。若表示反馈把行分布
限制在 KL 信任域 `epsilon` 内，则反馈造成的重心偏差最多是
`O(sqrt(epsilon))` 量级，而不是任意漂移。

边界：

- 这是幅度上界，不保证反馈方向正确；
- `W*` 不可观测时不能把经验结果解释成真实假负标签误差；
- 该命题不能替代下游检测实验，只能解释 KL 信任域为什么限制最坏漂移。

## 9. 写作和实现修正

- 画像统一写成“训练期行为画像先验（CERT 47 维，CLUE-LDS 105 维）”，不再把 47 维
  当成方法常量；
- A0/A1 入口、A3/A4 两阶段、EMA 教师和 Pinsker 命题分别标注实现状态；
- smoke、实现失败、统计功效不足和方法机制不支持必须使用状态码；
- 旧 20-pair CLUE 结果只能作为开发记录，不能进入 canonical 表。

## 10. 下一顺序

1. A0-A3batch、A1-Cov-A5a、A1-Cov-uniform 的入口和单元测试已接入；仍需在正式数据规模和正式矩阵中验证；
2. 冻结事件/用户簇功效模拟，并决定 r5.2 Gate；
3. 完成教师质量门禁和距离 clip 统计；
4. 再跑 CLUE-LDS 两 seed pilot；
5. 在结果出来后只允许按新勘误改变分析，不得改写 v1.2.7 的主比较。
